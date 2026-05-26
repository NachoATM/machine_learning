from __future__ import annotations

import logging
from collections import Counter
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.base import BaseCaso
from src.viz.plotter import Plotter

logger = logging.getLogger(__name__)


class EDACorpus(BaseCaso):
    """
    Caso 1 — Radiografía del Corpus Documental.

    Ejecuta las 9 figuras del análisis exploratorio original refactorizadas
    como métodos privados, más el clustering documental con TF-IDF + KMeans.
    """

    def __init__(self, df: pd.DataFrame, output_dir="outputs", fig_dir="figuras") -> None:
        super().__init__(df, output_dir, fig_dir)
        self.plotter = Plotter(fig_dir)
        self._top_personas: list[tuple[str, int]] = []
        self._top_lugares: list[tuple[str, int]] = []
        self._df_cluster: pd.DataFrame = pd.DataFrame()
        self._k_optimo: int = 0

    # ── API pública ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        logger.info("=== Caso 1: EDA Corpus ===")
        self._calcular_entidades()
        self._fig_distribucion_fuente()        # Fig 1A
        self._fig_paginas()                    # Fig 1B
        self._fig_distribucion_periodo()       # Fig 1C
        self._fig_extension_riqueza()          # Fig 2
        self._fig_timeline()                   # Fig 3
        self._fig_personas_top()               # Fig 4A
        self._fig_lugares_top()                # Fig 4B
        self._fig_actores_periodo()            # Fig 5
        self._fig_wordclouds_periodo()         # Fig 6
        self._calcular_clustering()
        self._fig_elbow_silhouette()           # Fig 7
        self._fig_pca2d()                      # Fig 8
        self._fig_perfil_clusters()            # Fig 9

        self._results = {
            "total_documentos": len(self.df),
            "documentos_con_ocr": int((self.df["n_palabras_ocr"] > 50).sum()),
            "total_palabras": int(self.df["n_palabras_ocr"].sum()),
            "ministerio_predominante": self.df["fuente"].value_counts().idxmax(),
            "periodo_dominante": self.df["periodo"].value_counts().idxmax(),
            "persona_top": self._top_personas[0][0].title() if self._top_personas else "N/A",
            "lugar_top": self._top_lugares[0][0].title() if self._top_lugares else "N/A",
            "clusters_k": self._k_optimo,
        }
        return self._results

    def export(self) -> None:
        self._save_json("resumen_caso1.json", self._results)
        if not self._df_cluster.empty:
            cols = ["id", "titulo", "fuente", "periodo", "cluster"]
            self._save_csv("clusters_caso1.csv", self._df_cluster[cols])

    # ── Figuras ────────────────────────────────────────────────────────────────

    def _fig_distribucion_fuente(self) -> None:
        conteo = self.df["fuente"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colores = [self.plotter.paleta_fuente.get(f, "#888") for f in conteo.index]
        ax.barh(conteo.index, conteo.values, color=colores, edgecolor="white")
        for i, v in enumerate(conteo.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=10)
        ax.set_title("Figura 1A — Documentos por Ministerio", fontweight="bold")
        ax.set_xlabel("N documentos")
        fig.tight_layout()
        self._savefig("fig1a_fuente.png", fig)

    def _fig_paginas(self) -> None:
        df_pag = self.df.dropna(subset=["paginas"])
        fig, ax = plt.subplots(figsize=(9, 4))
        for fuente, grupo in df_pag.groupby("fuente"):
            ax.hist(grupo["paginas"], bins=15, alpha=0.6,
                    color=self.plotter.paleta_fuente.get(fuente, "#888"),
                    label=fuente, edgecolor="white")
        ax.set_title("Figura 1B — Distribución de Páginas por Ministerio", fontweight="bold")
        ax.set_xlabel("Número de páginas")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._savefig("fig1b_paginas.png", fig)

    def _fig_distribucion_periodo(self) -> None:
        orden = list(self.plotter.paleta_periodo.keys())
        conteo = self.df["periodo"].value_counts().reindex(orden).dropna()
        colores = [self.plotter.paleta_periodo[p] for p in conteo.index]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(conteo.index, conteo.values, color=colores, edgecolor="white")
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
        ax.set_title("Figura 1C — Documentos por Período Histórico", fontweight="bold")
        ax.set_xlabel("Período")
        ax.set_ylabel("N documentos")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        self._savefig("fig1c_periodo.png", fig)

    def _fig_extension_riqueza(self) -> None:
        df_t = self.df[self.df["n_palabras_ocr"] > 0]
        fuentes = [f for f in self.plotter.paleta_fuente if f in df_t["fuente"].unique()]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Figura 2 — Extensión Textual y Riqueza Léxica", fontweight="bold")

        grupos = [df_t[df_t["fuente"] == f]["n_palabras_ocr"].values for f in fuentes]
        bp = axes[0].boxplot(grupos, labels=fuentes, patch_artist=True)
        for patch, f in zip(bp["boxes"], fuentes):
            patch.set_facecolor(self.plotter.paleta_fuente[f])
            patch.set_alpha(0.7)
        axes[0].set_title("A — Palabras OCR por Ministerio")
        axes[0].set_ylabel("N palabras")

        df_rl = self.df[self.df["riqueza_lexica"].notna() & (self.df["n_palabras_ocr"] > 50)]
        f_rl = [f for f in fuentes if f in df_rl["fuente"].unique()]
        datos_rl = [df_rl[df_rl["fuente"] == f]["riqueza_lexica"].values for f in f_rl]
        if datos_rl:
            parts = axes[1].violinplot(datos_rl, positions=range(len(f_rl)),
                                        showmeans=True, showmedians=True)
            for pc, f in zip(parts["bodies"], f_rl):
                pc.set_facecolor(self.plotter.paleta_fuente[f])
                pc.set_alpha(0.7)
        axes[1].set_xticks(range(len(f_rl)))
        axes[1].set_xticklabels(f_rl)
        axes[1].set_title("B — Riqueza Léxica (TTR)")
        axes[1].set_ylabel("Type-Token Ratio")
        fig.tight_layout()
        self._savefig("fig2_extension_riqueza.png", fig)

    def _fig_timeline(self) -> None:
        df_t = self.df[self.df["anio"].notna()].copy()
        df_t["anio"] = df_t["anio"].astype(int)
        pivot = (df_t.groupby(["anio", "fuente"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=list(self.plotter.paleta_fuente), fill_value=0))
        fig, ax = plt.subplots(figsize=(13, 5))
        bottom = np.zeros(len(pivot))
        for fuente, color in self.plotter.paleta_fuente.items():
            if fuente in pivot.columns:
                ax.bar(pivot.index, pivot[fuente], bottom=bottom,
                       color=color, label=fuente, alpha=0.85,
                       edgecolor="white", width=0.6)
                bottom += pivot[fuente].values
        ax.axvline(1981, color="black", linestyle="--", linewidth=1.8, alpha=0.8)
        ax.text(1981.1, ax.get_ylim()[1] * 0.88, "23-F\nfeb.1981",
                fontsize=9, color="black", style="italic")
        ax.set_title("Figura 3 — Distribución Temporal del Corpus", fontweight="bold")
        ax.set_xlabel("Año")
        ax.set_ylabel("N documentos")
        ax.set_xticks(sorted(df_t["anio"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=10)
        fig.tight_layout()
        self._savefig("fig3_timeline.png", fig)

    def _calcular_entidades(self) -> None:
        def norm(s: str) -> str:
            return s.strip().lower()

        self._top_personas = Counter(
            norm(p) for p in chain.from_iterable(self.df["personas"])
        ).most_common(30)
        self._top_lugares = Counter(
            norm(l) for l in chain.from_iterable(self.df["lugares"])
        ).most_common(25)

    def _fig_personas_top(self) -> None:
        if not self._top_personas:
            return
        nombres, counts = zip(*self._top_personas[:20])
        colores = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(nombres)))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(nombres)), counts, color=colores)
        ax.set_yticks(range(len(nombres)))
        ax.set_yticklabels([n.title() for n in nombres], fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Figura 4A — Personas más Mencionadas (Top 20)", fontweight="bold")
        ax.set_xlabel("N documentos")
        fig.tight_layout()
        self._savefig("fig4a_personas_top.png", fig)

    def _fig_lugares_top(self) -> None:
        if not self._top_lugares:
            return
        freq = dict(self._top_lugares[:60])
        wc = WordCloud(width=800, height=450, background_color="white",
                       colormap="Blues", max_words=60).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Figura 4B — Nube de Lugares Mencionados", fontweight="bold")
        fig.tight_layout()
        self._savefig("fig4b_lugares_wordcloud.png", fig)

    def _fig_actores_periodo(self) -> None:
        top12 = [p for p, _ in self._top_personas[:12]]
        periodos = ["Pre-golpe", "23-F (1981)", "Proceso judicial", "Post-proceso"]
        matriz = pd.DataFrame(0, index=top12, columns=periodos)
        for _, row in self.df.iterrows():
            if row["periodo"] not in periodos:
                continue
            for persona in (row["personas"] or []):
                p = persona.strip().lower()
                if p in top12:
                    matriz.loc[p, row["periodo"]] += 1
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white", ax=ax,
                    cbar_kws={"label": "N documentos"})
        ax.set_yticklabels([n.title() for n in matriz.index], rotation=0, fontsize=9)
        ax.set_xticklabels(periodos, rotation=15)
        ax.set_title("Figura 5 — Actores Clave × Período Histórico", fontweight="bold")
        fig.tight_layout()
        self._savefig("fig5_actores_periodo.png", fig)

    def _fig_wordclouds_periodo(self) -> None:
        colormaps = {"Pre-golpe": "Oranges", "23-F (1981)": "Reds",
                     "Proceso judicial": "Blues"}
        periodos_validos = list(colormaps.keys())
        corpus = {
            p: " ".join(
                self.plotter.limpiar_texto(t)
                for t in self.df[self.df["periodo"] == p]["texto_ocr"].dropna()
                if len(t) > 100
            )
            for p in periodos_validos
        }
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1, max_df=0.95)
        textos = list(corpus.values())
        if not any(textos):
            return
        mat = vec.fit_transform(textos)
        feats = vec.get_feature_names_out()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Figura 6 — Vocabulario Discriminante por Período (TF-IDF)", fontweight="bold")
        for ax, periodo, cmap in zip(axes, periodos_validos, colormaps.values()):
            row = mat[periodos_validos.index(periodo)].toarray().flatten()
            top_idx = row.argsort()[::-1][:30]
            freq = {feats[j]: float(row[j]) for j in top_idx if row[j] > 0}
            if freq:
                wc = WordCloud(width=550, height=350, background_color="white",
                               colormap=cmap, max_words=30).generate_from_frequencies(freq)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(periodo, fontweight="bold",
                         color=self.plotter.paleta_periodo.get(periodo, "black"))
        fig.tight_layout()
        self._savefig("fig6_wordclouds_periodo.png", fig)

    def _calcular_clustering(self) -> None:
        df_c = self.df[self.df["n_palabras_ocr"] >= 50].copy().reset_index(drop=True)
        if len(df_c) < 4:
            logger.warning("Insuficientes documentos para clustering.")
            return

        textos = df_c["texto_ocr"].apply(self.plotter.limpiar_texto).tolist()
        vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2),
                              min_df=2, max_df=0.90, sublinear_tf=True)
        X = vec.fit_transform(textos)

        n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
        pca_pre = PCA(n_components=n_comp, random_state=42)
        X_red = pca_pre.fit_transform(X.toarray())
        logger.info("Varianza PCA(%d): %.1f%%", n_comp,
                    pca_pre.explained_variance_ratio_.sum() * 100)

        rango_k = range(2, min(9, len(df_c)))
        siluetas, inercias = [], []
        for k in rango_k:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiq = km.fit_predict(X_red)
            siluetas.append(silhouette_score(X_red, etiq))
            inercias.append(km.inertia_)

        self._k_optimo = list(rango_k)[int(np.argmax(siluetas))]
        self._siluetas = siluetas
        self._inercias = inercias
        self._rango_k = list(rango_k)

        km_final = KMeans(n_clusters=self._k_optimo, random_state=42, n_init=15)
        df_c["cluster"] = km_final.fit_predict(X_red)

        pca2 = PCA(n_components=2, random_state=42)
        X_2d = pca2.fit_transform(X_red)
        df_c["pca1"] = X_2d[:, 0]
        df_c["pca2"] = X_2d[:, 1]
        self._pca2_var = pca2.explained_variance_ratio_

        feats = vec.get_feature_names_out()
        self._top_terms_cluster: dict[int, list[str]] = {}
        for c in range(self._k_optimo):
            idx = df_c[df_c["cluster"] == c].index.tolist()
            if idx:
                centroide = X[idx].mean(axis=0).A1
                top_idx = centroide.argsort()[::-1][:10]
                self._top_terms_cluster[c] = [feats[i] for i in top_idx]

        self._df_cluster = df_c
        self._X_tfidf = X

    def _fig_elbow_silhouette(self) -> None:
        if not hasattr(self, "_siluetas"):
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle("Figura 7 — Selección del K Óptimo", fontweight="bold")
        axes[0].plot(self._rango_k, self._inercias, "o-", color="#2166ac", linewidth=2)
        axes[0].axvline(self._k_optimo, color="red", linestyle="--", alpha=0.7,
                        label=f"k={self._k_optimo}")
        axes[0].set_title("A — Curva del Codo (Inercia)")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inercia")
        axes[0].legend()
        axes[1].plot(self._rango_k, self._siluetas, "s-", color="#d73027", linewidth=2)
        axes[1].axvline(self._k_optimo, color="red", linestyle="--", alpha=0.7,
                        label=f"k={self._k_optimo}")
        axes[1].set_title("B — Silhouette Score")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silhouette")
        axes[1].legend()
        fig.tight_layout()
        self._savefig("fig7_elbow_silhouette.png", fig)

    def _fig_pca2d(self) -> None:
        if self._df_cluster.empty:
            return
        palette = plt.cm.tab10(np.linspace(0, 0.9, self._k_optimo))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Figura 8 — Clustering TF-IDF + PCA 2D", fontweight="bold")

        for c in range(self._k_optimo):
            mask = self._df_cluster["cluster"] == c
            label = f"C{c}: " + ", ".join(self._top_terms_cluster.get(c, [])[:3])
            axes[0].scatter(self._df_cluster.loc[mask, "pca1"],
                            self._df_cluster.loc[mask, "pca2"],
                            c=[palette[c]], label=label,
                            alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
        axes[0].set_title("A — Clusters K-Means")
        axes[0].set_xlabel(f"PC1 ({self._pca2_var[0]:.1%} var.)")
        axes[0].set_ylabel(f"PC2 ({self._pca2_var[1]:.1%} var.)")
        axes[0].legend(fontsize=7)

        for fuente, color in self.plotter.paleta_fuente.items():
            mask = self._df_cluster["fuente"] == fuente
            if mask.any():
                axes[1].scatter(self._df_cluster.loc[mask, "pca1"],
                                self._df_cluster.loc[mask, "pca2"],
                                c=color, label=fuente,
                                alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
        axes[1].set_title("B — Fuente Real (Ministerio)")
        axes[1].set_xlabel(f"PC1 ({self._pca2_var[0]:.1%} var.)")
        axes[1].set_ylabel(f"PC2 ({self._pca2_var[1]:.1%} var.)")
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        self._savefig("fig8_pca2d_clusters.png", fig)

    def _fig_perfil_clusters(self) -> None:
        if self._df_cluster.empty:
            return
        feats = ["paginas", "n_personas", "n_lugares", "n_palabras_ocr",
                 "riqueza_lexica", "densidad_entidades", "compresion_resumen"]
        perfil = (self._df_cluster.groupby("cluster")[feats]
                  .mean().dropna(axis=1, how="all"))
        norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.heatmap(norm.T, annot=perfil.T.round(2), fmt="g",
                    cmap="RdYlGn", linewidths=0.5, linecolor="white", ax=ax,
                    cbar_kws={"label": "Valor normalizado [0-1]"})
        ax.set_xticklabels([f"Cluster {c}" for c in perfil.index])
        ax.set_title("Figura 9 — Perfil de Features por Cluster", fontweight="bold")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        self._savefig("fig9_perfil_clusters.png", fig)