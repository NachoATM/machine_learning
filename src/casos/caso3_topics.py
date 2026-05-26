"""TopicModelCaso: topic modeling semántico con BERTopic"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from src.base import BaseCaso
from src.viz.plotter import Plotter

logger = logging.getLogger(__name__)

# ── Imports opcionales ─────────────────────────────────────────────────────────
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    _HAS_BERTOPIC = True
except ImportError:
    _HAS_BERTOPIC = False
    logger.info("BERTopic/sentence-transformers no disponible — usando NMF como fallback.")

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    logger.info("umap-learn no disponible — se usará PCA para visualización 2D.")


class TopicModelCaso(BaseCaso):
    """
    Caso 3 — Topic Modeling Semántico.

    Fallback robusto a NMF sobre TF-IDF con la misma interfaz de resultados.
    Incluye análisis de tópicos por período, detección de documentos ambiguos
    (alta entropía) y comparación con los clusters del Caso 1 via ARI.
    """

    _MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, df: pd.DataFrame, output_dir="outputs", fig_dir="figuras",
                 min_topic_size: int = 3) -> None:
        super().__init__(df, output_dir, fig_dir)
        self.plotter = Plotter(fig_dir)
        self.min_topic_size = min_topic_size
        self._topic_labels: list[int] = []
        self._topic_words: dict[int, list[str]] = {}
        self._topic_word_scores: dict[int, list[float]] = {}
        self._doc_topic_probs: np.ndarray | None = None
        self._outlier_docs: pd.DataFrame = pd.DataFrame()
        self._topic_summary: pd.DataFrame = pd.DataFrame()
        self._projection_2d: np.ndarray | None = None
        self._projection_method: str = "N/A"

    # ── API pública ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        logger.info("=== Caso 3: Topic Modeling ===")
        df_valid = self.df[self.df["n_palabras_ocr"] >= 50].copy().reset_index(drop=True)
        textos = df_valid["texto_ocr"].apply(self.plotter.limpiar_texto).tolist()

        if _HAS_BERTOPIC:
            self._run_bertopic(textos, df_valid)
        else:
            self._run_nmf(textos, df_valid)

        df_valid["topic"] = self._topic_labels
        self._df_valid = df_valid

        self._outlier_docs = self._get_outlier_docs(df_valid)
        self._topic_summary = self._build_topic_summary(df_valid)
        self._fig_top_words_por_topico()        # Fig 3-1
        self._fig_topicos_por_periodo(df_valid) # Fig 3-2
        self._fig_scatter_2d(textos, df_valid)  # Fig 3-3
        self._html_panel_interactivo()          # Fig 3-4

        # ARI vs clusters Caso 1 (si existe la columna cluster)
        ari = np.nan
        if "cluster" in df_valid.columns:
            mask = df_valid["cluster"].notna() & df_valid["topic"].notna()
            if mask.sum() > 1:
                ari = adjusted_rand_score(
                    df_valid.loc[mask, "cluster"].astype(int),
                    df_valid.loc[mask, "topic"].astype(int),
                )

        panel_html = self.fig_dir / "fig3_4_panel_topicos_interactivo.html"
        self._results = {
            "backend": "BERTopic" if _HAS_BERTOPIC else "NMF",
            "n_topicos": len(self._topic_words),
            "n_documentos": len(df_valid),
            "n_outliers": len(self._outlier_docs),
            "ari_vs_caso1": round(float(ari), 4) if not np.isnan(ari) else "N/A",
            "topico_mayor": (
                int(self._topic_summary.iloc[0]["topic"])
                if not self._topic_summary.empty else "N/A"
            ),
            "panel_interactivo": str(panel_html) if panel_html.exists() else "N/A",
        }
        return self._results

    def export(self) -> None:
        if hasattr(self, "_df_valid") and not self._df_valid.empty:
            cols = ["id", "titulo", "fuente", "periodo", "topic"]
            if self._doc_topic_probs is not None:
                self._df_valid = self._df_valid.copy()
                self._df_valid["prob_topic"] = self._doc_topic_probs.max(axis=1)
                self._df_valid["entropia_topico"] = self._entropy_scores()
                cols.extend(["prob_topic", "entropia_topico"])
            self._save_csv("topicos_por_documento.csv", self._df_valid[cols])
        if not self._topic_summary.empty:
            self._save_csv("resumen_topicos_caso3.csv", self._topic_summary)
        if not self._outlier_docs.empty:
            self._save_csv("outliers_caso3.csv", self._outlier_docs)

    # ── Métodos de análisis ────────────────────────────────────────────────────

    def fit(self, min_topic_size: int | None = None) -> None:
        """Permite re-ajustar el modelo con distintos parámetros."""
        if min_topic_size is not None:
            self.min_topic_size = min_topic_size
        self.run()

    def get_topics_per_period(self) -> pd.DataFrame:
        """Distribución de tópicos por período histórico."""
        if not hasattr(self, "_df_valid"):
            return pd.DataFrame()
        return (self._df_valid.groupby(["periodo", "topic"])
                .size().unstack(fill_value=0))

    def get_outlier_docs(self) -> pd.DataFrame:
        return self._outlier_docs

    # ── Backends ───────────────────────────────────────────────────────────────

    def _run_bertopic(self, textos: list[str], df: pd.DataFrame) -> None:
        logger.info("Ajustando BERTopic con %s...", self._MODELO_EMBEDDINGS)
        embedder = SentenceTransformer(self._MODELO_EMBEDDINGS)
        embeddings = embedder.encode(textos, show_progress_bar=True)
        model = BERTopic(
            language="multilingual",
            min_topic_size=self.min_topic_size,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = model.fit_transform(textos, embeddings)
        self._topic_labels = topics
        self._doc_topic_probs = probs
        self._bertopic_model = model
        # Extraer palabras por tópico
        for tid in set(topics):
            if tid == -1:
                continue
            info = model.get_topic(tid)
            self._topic_words[tid] = [w for w, _ in info[:10]] if info else []
            self._topic_word_scores[tid] = [float(score) for _, score in info[:10]] if info else []
        logger.info("BERTopic: %d tópicos descubiertos", len(self._topic_words))

    def _run_nmf(self, textos: list[str], df: pd.DataFrame) -> None:
        logger.info("Ajustando NMF (fallback BERTopic)...")
        n_topics = max(4, min(9, len(textos) // 12))
        vec = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.86,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words=list(self._STOPWORDS_ES),
        )
        X = vec.fit_transform(textos)
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=600,
            init="nndsvda",
            l1_ratio=0.15,
        )
        W = model.fit_transform(X)  # doc × topic
        H = model.components_       # topic × term
        feats = vec.get_feature_names_out()
        self._doc_topic_probs = normalize(W, norm="l1")
        self._topic_labels = list(W.argmax(axis=1))
        for tid in range(n_topics):
            top_idx = H[tid].argsort()[::-1][:12]
            self._topic_words[tid] = [feats[i] for i in top_idx]
            self._topic_word_scores[tid] = [float(H[tid, i]) for i in top_idx]
        logger.info("NMF: %d tópicos", n_topics)

    # ── Outliers ───────────────────────────────────────────────────────────────

    def _get_outlier_docs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Documentos con alta entropía de probabilidades significa contenido ambiguo."""
        if self._doc_topic_probs is None:
            return pd.DataFrame()
        entropia_norm = self._entropy_scores()
        df = df.copy()
        df["entropia_topico"] = entropia_norm
        df["prob_topic"] = self._doc_topic_probs.max(axis=1)
        umbral = np.percentile(entropia_norm, 75)
        outliers = df[df["entropia_topico"] > umbral].sort_values(
            "entropia_topico", ascending=False
        ).head(10)
        return outliers[["id", "titulo", "fuente", "periodo", "topic", "prob_topic", "entropia_topico"]]

    def _entropy_scores(self) -> np.ndarray:
        if self._doc_topic_probs is None:
            return np.array([])
        probs = self._doc_topic_probs
        eps = 1e-10
        entropia = -np.sum(probs * np.log2(probs + eps), axis=1)
        return entropia / np.log2(probs.shape[1] + 1)

    def _build_topic_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._doc_topic_probs is None or "topic" not in df.columns:
            return pd.DataFrame()
        rows = []
        prob_max = self._doc_topic_probs.max(axis=1)
        for tid in sorted(t for t in set(self._topic_labels) if t != -1):
            mask = df["topic"].to_numpy() == tid
            if not mask.any():
                continue
            periodos = df.loc[mask, "periodo"].value_counts()
            fuentes = df.loc[mask, "fuente"].value_counts()
            rep_idx = np.where(mask)[0][np.argmax(prob_max[mask])]
            rows.append({
                "topic": int(tid),
                "n_documentos": int(mask.sum()),
                "peso_corpus": round(float(mask.mean()), 4),
                "periodo_principal": periodos.index[0] if not periodos.empty else "N/A",
                "fuente_principal": fuentes.index[0] if not fuentes.empty else "N/A",
                "prob_media": round(float(prob_max[mask].mean()), 4),
                "palabras_clave": ", ".join(self._topic_words.get(tid, [])[:8]),
                "documento_representativo": df.iloc[rep_idx]["titulo"],
            })
        return pd.DataFrame(rows).sort_values("n_documentos", ascending=False).reset_index(drop=True)

    # ── Figuras ────────────────────────────────────────────────────────────────

    def _fig_top_words_por_topico(self) -> None:
        if not self._topic_words:
            return
        topics_to_plot = sorted(self._topic_words)[:8]
        n_cols = min(4, len(topics_to_plot))
        n_rows = (len(topics_to_plot) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle("Figura 3-1 — Top-10 Palabras por Tópico", fontweight="bold")
        palette = plt.cm.tab10(np.linspace(0, 0.9, len(topics_to_plot)))

        for ax, tid, color in zip(axes, topics_to_plot, palette):
            words = self._topic_words[tid][:10]
            if not words:
                ax.axis("off")
                continue
            scores = np.linspace(1, 0.3, len(words))
            ax.barh(range(len(words)), scores, color=color, alpha=0.8, edgecolor="white")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"Tópico {tid}", fontsize=9, fontweight="bold")
            ax.set_xlim(0, 1.1)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        for ax in axes[len(topics_to_plot):]:
            ax.axis("off")
        fig.tight_layout()
        self._savefig("fig3_1_topwords_topico.png", fig)

    def _fig_topicos_por_periodo(self, df: pd.DataFrame) -> None:
        pivot = self.get_topics_per_period()
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.3, linecolor="white", ax=ax,
                    cbar_kws={"label": "N documentos"})
        ax.set_title("Figura 3-2 — Distribución de Tópicos respecto al Período Histórico",
                     fontweight="bold")
        ax.set_xlabel("Tópico")
        ax.set_ylabel("Período")
        fig.tight_layout()
        self._savefig("fig3_2_topicos_periodo.png", fig)

    def _fig_scatter_2d(self, textos: list[str], df: pd.DataFrame) -> None:
        """Scatter UMAP (fallback PCA) coloreado por tópico."""
        X_2d, metodo = self._compute_projection_2d(textos)
        if X_2d is None:
            return

        self._projection_2d = X_2d
        self._projection_method = metodo
        topics_arr = np.array(self._topic_labels)
        unique_topics = sorted(set(topics_arr))
        palette = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))
        color_map = {t: palette[i] for i, t in enumerate(unique_topics)}

        fig, ax = plt.subplots(figsize=(12, 7))
        for tid in unique_topics:
            mask = topics_arr == tid
            label = f"T{tid}: " + ", ".join(self._topic_words.get(tid, ["?"])[:2])
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color_map[tid]], label=label,
                       alpha=0.75, s=50, edgecolors="white", linewidths=0.3)
        ax.set_title(f"Figura 3-3 — Documentos en espacio {metodo} coloreados por Tópico",
                     fontweight="bold")
        ax.set_xlabel(f"{metodo} 1")
        ax.set_ylabel(f"{metodo} 2")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        self._savefig("fig3_3_scatter_topicos.png", fig)

    def _compute_projection_2d(self, textos: list[str]) -> tuple[np.ndarray | None, str]:
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            max_features=700,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            stop_words=list(self._STOPWORDS_ES),
        )
        try:
            X = vec.fit_transform(textos).toarray()
        except Exception:
            return None, "N/A"

        if _HAS_UMAP and X.shape[0] > 5:
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10)
                X_2d = reducer.fit_transform(X)
                metodo = "UMAP"
            except Exception:
                X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
                metodo = "PCA"
        else:
            X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
            metodo = "PCA"
        return X_2d, metodo

    # ── Panel interactivo ─────────────────────────────────────────────────────

    def _html_panel_interactivo(self) -> None:
        if not hasattr(self, "_df_valid") or self._df_valid.empty:
            return
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly no disponible — no se genera el panel interactivo.")
            return

        df = self._df_valid.copy()
        if self._doc_topic_probs is not None:
            df["prob_topic"] = self._doc_topic_probs.max(axis=1)
            df["entropia_topico"] = self._entropy_scores()
        else:
            df["prob_topic"] = np.nan
            df["entropia_topico"] = np.nan

        if self._projection_2d is None:
            textos = df["texto_ocr"].apply(self.plotter.limpiar_texto).tolist()
            self._projection_2d, self._projection_method = self._compute_projection_2d(textos)
        if self._projection_2d is None:
            return

        df["x"] = self._projection_2d[:, 0]
        df["y"] = self._projection_2d[:, 1]
        topics = sorted(t for t in set(self._topic_labels) if t != -1)
        topic_counts = df["topic"].value_counts().sort_index()
        period_topic = self.get_topics_per_period()
        entropy = df["entropia_topico"].dropna()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Documentos en espacio {self._projection_method}",
                "Distribución tópico-periodo",
                "Tamaño de los tópicos",
                "Ambigüedad temática",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "histogram"}],
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        for tid in topics:
            mask = df["topic"] == tid
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "x"],
                    y=df.loc[mask, "y"],
                    mode="markers",
                    name=f"T{tid}",
                    marker=dict(size=10, line=dict(width=0.6, color="#ffffff"), opacity=0.82),
                    customdata=np.stack([
                        df.loc[mask, "id"].astype(str),
                        df.loc[mask, "titulo"].fillna("").astype(str).str.slice(0, 120),
                        df.loc[mask, "periodo"].fillna("").astype(str),
                        df.loc[mask, "fuente"].fillna("").astype(str),
                        df.loc[mask, "prob_topic"].fillna(0).round(3),
                        df.loc[mask, "entropia_topico"].fillna(0).round(3),
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        "Tópico: T" + str(tid) + "<br>"
                        "Período: %{customdata[2]}<br>"
                        "Fuente: %{customdata[3]}<br>"
                        "Probabilidad: %{customdata[4]}<br>"
                        "Entropía: %{customdata[5]}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Heatmap(
                z=period_topic.values,
                x=[f"T{c}" for c in period_topic.columns],
                y=period_topic.index,
                colorscale="Blues",
                text=period_topic.values,
                texttemplate="%{text}",
                hovertemplate="Período: %{y}<br>Tópico: %{x}<br>Documentos: %{z}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=[f"T{i}" for i in topic_counts.index],
                y=topic_counts.values,
                marker_color="#2166ac",
                hovertemplate="%{x}<br>Documentos: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=entropy,
                nbinsx=20,
                marker_color="#d95f02",
                hovertemplate="Entropía: %{x:.3f}<br>Documentos: %{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Figura 3-4 — Panel Interactivo de Tópicos",
            height=900,
            template="plotly_white",
            margin=dict(l=60, r=40, t=90, b=50),
            legend_title_text="Tópico",
        )
        fig.update_xaxes(title_text=f"{self._projection_method} 1", row=1, col=1)
        fig.update_yaxes(title_text=f"{self._projection_method} 2", row=1, col=1)
        fig.update_xaxes(title_text="Tópico", row=2, col=1)
        fig.update_yaxes(title_text="Documentos", row=2, col=1)
        fig.update_xaxes(title_text="Entropía normalizada", row=2, col=2)
        fig.update_yaxes(title_text="Documentos", row=2, col=2)

        html_fig = fig.to_html(full_html=False, include_plotlyjs=True)
        contenido = self._envolver_panel_html(html_fig)
        path = self.fig_dir / "fig3_4_panel_topicos_interactivo.html"
        path.write_text(contenido, encoding="utf-8")
        logger.info("Panel interactivo guardado: %s", path)

    def _tabla_topicos_html(self) -> str:
        if self._topic_summary.empty:
            return ""
        df = self._topic_summary.copy()
        df["documento_representativo"] = df["documento_representativo"].astype(str).str.slice(0, 120)
        for col in ["palabras_clave", "documento_representativo"]:
            df[col] = df[col].map(lambda x: html.escape(str(x)))
        return df.to_html(index=False, classes="tabla", border=0, table_id="topicos")

    def _tabla_documentos_html(self) -> str:
        if not hasattr(self, "_df_valid") or self._df_valid.empty:
            return ""
        df = self._df_valid[["id", "titulo", "fuente", "periodo", "topic"]].copy()
        if self._doc_topic_probs is not None:
            df["prob_topic"] = self._doc_topic_probs.max(axis=1).round(3)
            df["entropia_topico"] = self._entropy_scores().round(3)
        df["titulo"] = df["titulo"].fillna("").astype(str).str.slice(0, 140)
        return df.sort_values(["topic", "prob_topic"], ascending=[True, False]).to_html(
            index=False, classes="tabla", border=0, table_id="documentos"
        )
    

    def _envolver_panel_html(self, grafico: str) -> str:
        topicos = self._tabla_topicos_html()
        documentos = self._tabla_documentos_html()
        return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panel interactivo caso 3</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f6f8fa;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 16px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 17px; }}
    input {{
      width: min(520px, 100%);
      padding: 9px 11px;
      border: 1px solid #b8c2cc;
      border-radius: 6px;
      font-size: 14px;
      margin-bottom: 12px;
    }}
    .tabla {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .tabla th {{
      text-align: left;
      background: #eef2f7;
      position: sticky;
      top: 0;
    }}
    .tabla th, .tabla td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e1e6ee;
      vertical-align: top;
    }}
    .table-wrap {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid #e1e6ee;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Panel interactivo de tópicos</h1>
      {grafico}
    </section>
    <section>
      <h2>Resumen de tópicos</h2>
      <input id="filtroTopicos" type="search" placeholder="Filtrar por tópico, período, palabras o documento">
      <div class="table-wrap">{topicos}</div>
    </section>
    <section>
      <h2>Documentos etiquetados</h2>
      <input id="filtroDocs" type="search" placeholder="Filtrar por id, título, fuente, período o tópico">
      <div class="table-wrap">{documentos}</div>
    </section>
  </main>
  <script>
    function activarFiltro(inputId, tableId) {{
      const filtro = document.getElementById(inputId);
      const tabla = document.getElementById(tableId);
      if (!filtro || !tabla) return;
      const filas = Array.from(tabla.querySelectorAll("tbody tr"));
      filtro.addEventListener("input", () => {{
        const q = filtro.value.toLowerCase();
        filas.forEach((fila) => {{
          fila.style.display = fila.innerText.toLowerCase().includes(q) ? "" : "none";
        }});
      }});
    }}
    activarFiltro("filtroTopicos", "topicos");
    activarFiltro("filtroDocs", "documentos");
  </script>
</body>
</html>"""

    _STOPWORDS_ES = frozenset({
        "a", "al", "algo", "ante", "antes", "aquel", "aquella", "aquellas",
        "aquello", "aquellos", "aqui", "asi", "cada", "como", "con", "contra",
        "cual", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella",
        "ellas", "ello", "ellos", "en", "entre", "era", "eran", "es", "esa",
        "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado",
        "estan", "estar", "estas", "este", "esto", "estos", "fue", "fueron",
        "ha", "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los",
        "mas", "me", "mi", "muy", "no", "nos", "o", "para", "pero", "por",
        "porque", "que", "se", "ser", "si", "sin", "sobre", "son", "su",
        "sus", "tambien", "te", "tiene", "tienen", "todo", "tras", "un",
        "una", "uno", "unos", "y", "ya",
        "dia", "dias", "fecha", "folio", "folios", "hora", "horas", "nota",
        "pag", "pagina", "punto", "segun", "sesion", "siguiente",
    })
