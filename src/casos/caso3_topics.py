"""TopicModelCaso: topic modeling semántico con BERTopic"""

from __future__ import annotations

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
        self._doc_topic_probs: np.ndarray | None = None
        self._outlier_docs: pd.DataFrame = pd.DataFrame()

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
        self._fig_top_words_por_topico()        # Fig 3-1
        self._fig_topicos_por_periodo(df_valid) # Fig 3-2
        self._fig_scatter_2d(textos, df_valid)  # Fig 3-3

        # ARI vs clusters Caso 1 (si existe la columna cluster)
        ari = np.nan
        if "cluster" in df_valid.columns:
            mask = df_valid["cluster"].notna() & df_valid["topic"].notna()
            if mask.sum() > 1:
                ari = adjusted_rand_score(
                    df_valid.loc[mask, "cluster"].astype(int),
                    df_valid.loc[mask, "topic"].astype(int),
                )

        self._results = {
            "backend": "BERTopic" if _HAS_BERTOPIC else "NMF",
            "n_topicos": len(self._topic_words),
            "n_documentos": len(df_valid),
            "n_outliers": len(self._outlier_docs),
            "ari_vs_caso1": round(float(ari), 4) if not np.isnan(ari) else "N/A",
        }
        return self._results

    def export(self) -> None:
        if hasattr(self, "_df_valid") and not self._df_valid.empty:
            cols = ["id", "titulo", "fuente", "periodo", "topic"]
            self._save_csv("topicos_por_documento.csv", self._df_valid[cols])
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
        logger.info("BERTopic: %d tópicos descubiertos", len(self._topic_words))

    def _run_nmf(self, textos: list[str], df: pd.DataFrame) -> None:
        logger.info("Ajustando NMF (fallback BERTopic)...")
        n_topics = max(3, min(8, len(textos) // 10))
        vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2),
                              min_df=2, max_df=0.90, sublinear_tf=True)
        X = vec.fit_transform(textos)
        model = NMF(n_components=n_topics, random_state=42, max_iter=400)
        W = model.fit_transform(X)  # doc × topic
        H = model.components_       # topic × term
        feats = vec.get_feature_names_out()
        self._doc_topic_probs = normalize(W, norm="l1")
        self._topic_labels = list(W.argmax(axis=1))
        for tid in range(n_topics):
            top_idx = H[tid].argsort()[::-1][:10]
            self._topic_words[tid] = [feats[i] for i in top_idx]
        logger.info("NMF: %d tópicos", n_topics)

    # ── Outliers ───────────────────────────────────────────────────────────────

    def _get_outlier_docs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Documentos con alta entropía de probabilidades significa contenido ambiguo."""
        if self._doc_topic_probs is None:
            return pd.DataFrame()
        probs = self._doc_topic_probs
        # Entropía de Shannon normalizada
        eps = 1e-10
        entropia = -np.sum(probs * np.log2(probs + eps), axis=1)
        entropia_norm = entropia / np.log2(probs.shape[1] + 1)
        df = df.copy()
        df["entropia_topico"] = entropia_norm
        umbral = np.percentile(entropia_norm, 75)
        outliers = df[df["entropia_topico"] > umbral].sort_values(
            "entropia_topico", ascending=False
        ).head(10)
        return outliers[["id", "titulo", "fuente", "periodo", "topic", "entropia_topico"]]

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
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=500, sublinear_tf=True, min_df=2)
        try:
            X = vec.fit_transform(textos).toarray()
        except Exception:
            return

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
