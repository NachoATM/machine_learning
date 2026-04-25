"""ActorGraphCaso: grafo de co-menciones de actores entre documentos del 23-F."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.base import BaseCaso
from src.viz.plotter import Plotter

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.warning("networkx no disponible — Caso 2 deshabilitado.")

try:
    import community as community_louvain  # python-louvain
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False
    logger.info("python-louvain no disponible — se usará greedy_modularity_communities.")

try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False
    logger.info("spaCy no disponible — usando columna 'personas' de RTVE.")


COMM_COLORS = ["#1A356E", "#C01C2C", "#2C6E49", "#E8A020", "#7B2D8B",
               "#0077B6", "#6D3A3A", "#3D6B61", "#8B5E00", "#3A3A6B"]

PERIODO_COLORS = {
    "Pre-golpe":        "#1A356E",
    "23-F (1981)":      "#C01C2C",
    "Proceso judicial": "#2C6E49",
    "Post-proceso":     "#E8A020",
    "Desconocido":      "#888888",
}


class ActorGraphCaso(BaseCaso):
    """
    Caso 2 — Grafo de Co-menciones de Actores.

    Construye un grafo donde los nodos son personas y las aristas representan
    co-apariciones en el mismo documento (peso = frecuencia). Calcula métricas
    de centralidad, detecta comunidades y genera subgrafos por período histórico.
    """

    def __init__(self, df: pd.DataFrame, output_dir="outputs", fig_dir="figuras") -> None:
        super().__init__(df, output_dir, fig_dir)
        self.plotter = Plotter(fig_dir)
        self._graph: "nx.Graph | None" = None
        self._centrality_df: pd.DataFrame = pd.DataFrame()
        self._partition: dict[str, int] = {}

    # ── API pública ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        if not _HAS_NX:
            logger.error("networkx requerido para Caso 2.")
            self._results = {"error": "networkx no instalado"}
            return self._results

        logger.info("=== Caso 2: Grafo de Actores ===")

        personas_por_doc = self._extraer_personas()
        self._graph = self._build_graph(personas_por_doc)
        logger.info("Grafo: %d nodos, %d aristas",
                    self._graph.number_of_nodes(), self._graph.number_of_edges())

        self._centrality_df = self.compute_centrality()
        self._partition = self.detect_communities()

        top5 = (
            self._centrality_df.nlargest(5, "pagerank")[["nodo", "pagerank"]]
            .to_string(index=False)
        )
        logger.info("Top 5 por PageRank:\n%s", top5)

        # ── Figuras ────────────────────────────────────────────────────────────
        self._fig_red_principal()       # Fig 2-1: comunidades + betweenness (2 subplots)
        self._fig_subgrafos_periodo()   # Fig 2-2: fila de subgrafos temporales

        self._results = {
            "n_nodos": self._graph.number_of_nodes(),
            "n_aristas": self._graph.number_of_edges(),
            "n_comunidades": len(set(self._partition.values())),
            "actor_pagerank_top": (
                self._centrality_df.iloc[0]["nodo"]
                if not self._centrality_df.empty else "N/A"
            ),
        }
        return self._results

    def export(self) -> None:
        if not self._centrality_df.empty:
            path = self.output_dir / "metricas_centralidad.csv"
            self._centrality_df.to_csv(path, index=False)
            logger.info("CSV guardado: %s", path)
        if _HAS_NX and self._graph is not None:
            path = self.output_dir / "grafo_23f.gexf"
            nx.write_gexf(self._graph, str(path))
            logger.info("GEXF guardado: %s", path)

    # ── Análisis ───────────────────────────────────────────────────────────────

    def compute_centrality(self) -> pd.DataFrame:
        """PageRank, betweenness y degree ponderados para cada nodo."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return pd.DataFrame()
        G = self._graph
        pagerank    = nx.pagerank(G, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        degree      = dict(G.degree(weight="weight"))
        records = [
            {
                "nodo":            n,
                "pagerank":        round(pagerank[n], 6),
                "betweenness":     round(betweenness[n], 6),
                "degree_weighted": degree[n],
                "n_docs":          G.nodes[n].get("n_docs", 0),
            }
            for n in G.nodes()
        ]
        return (
            pd.DataFrame(records)
            .sort_values("pagerank", ascending=False)
            .reset_index(drop=True)
        )

    def detect_communities(self) -> dict[str, int]:
        """Louvain si está disponible; si no, greedy modularity."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return {}
        if _HAS_LOUVAIN:
            return community_louvain.best_partition(
                self._graph, weight="weight", random_state=42
            )
        comps = nx.community.greedy_modularity_communities(self._graph, weight="weight")
        return {node: i for i, comp in enumerate(comps) for node in comp}

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _extraer_personas(self, df: pd.DataFrame | None = None) -> list[list[str]]:
        df = df if df is not None else self.df
        if _HAS_SPACY:
            return self._ner_spacy(df)
        # Fallback: columna 'personas' ya extraída por RTVE (lista de strings)
        return [
            [p.strip().lower() for p in (row.get("personas") or []) if p.strip()]
            for _, row in df.iterrows()
        ]

    def _ner_spacy(self, df: pd.DataFrame) -> list[list[str]]:
        try:
            nlp = spacy.load("es_core_news_lg")
        except OSError:
            logger.warning("Modelo es_core_news_lg no encontrado. Usando fallback RTVE.")
            return [
                [p.strip().lower() for p in (row.get("personas") or [])]
                for _, row in df.iterrows()
            ]
        result: list[list[str]] = []
        for texto in df["texto_ocr"].fillna(""):
            doc = nlp(texto[:100_000])
            result.append(
                [ent.text.strip().lower() for ent in doc.ents if ent.label_ == "PER"]
            )
        return result

    def _build_graph(self, personas_por_doc: list[list[str]]) -> "nx.Graph":
        G = nx.Graph()
        freq_global = Counter(p for doc in personas_por_doc for p in doc)
        for nodo, n_docs in freq_global.items():
            G.add_node(nodo, n_docs=n_docs)
        for personas in personas_por_doc:
            for a, b in combinations(set(personas), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
        # Eliminar nodos aislados (sin aristas)
        G.remove_nodes_from([n for n, d in dict(G.degree()).items() if d == 0])
        return G

    def _subgrafo_top(self, G: "nx.Graph", max_nodos: int = 60) -> "nx.Graph":
        """Recorta a los max_nodos con mayor degree para legibilidad."""
        if G.number_of_nodes() <= max_nodos:
            return G
        top = sorted(dict(G.degree(weight="weight")), key=lambda n: G.degree(n, weight="weight"), reverse=True)[:max_nodos]
        return G.subgraph(top).copy()

    def _label(self, nombre: str) -> str:
        """Apellido + inicial si el nombre es muy largo."""
        partes = nombre.strip().title().split()
        if len(partes) >= 2 and len(nombre) > 16:
            return f"{partes[-1]}, {partes[0][0]}."
        return nombre.title()

    def _node_sizes_from_degree(self, G: "nx.Graph", base: int = 400, scale: int = 200) -> list[float]:
        """Tamaño de nodo proporcional al degree ponderado (estilo run_caso1)."""
        degrees = dict(G.degree(weight="weight"))
        max_d = max(degrees.values()) if degrees else 1
        return [base + (degrees[n] / max_d) * scale for n in G.nodes()]

    # ── Figuras ────────────────────────────────────────────────────────────────

    def _fig_red_principal(self) -> None:
        """
        Figura 2-1: dos subplots lado a lado.
          Izquierda — comunidades coloreadas (nodo ∝ degree).
          Derecha   — betweenness centrality (colormap YlOrRd).
        Estilo idéntico al run_caso1 de referencia.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return

        G = self._subgrafo_top(self._graph, max_nodos=60)
        pos = nx.spring_layout(G, seed=42, k=2.5)

        id_col = "nodo"
        cent = self._centrality_df.set_index(id_col) if not self._centrality_df.empty else pd.DataFrame()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Red de actores del 23-F", fontsize=14, fontweight="bold")

        # ── Subplot izquierdo: comunidades ────────────────────────────────────
        node_colors_comm = [
            COMM_COLORS[self._partition.get(n, 0) % len(COMM_COLORS)]
            for n in G.nodes()
        ]
        if not cent.empty:
            sizes = [
                800 + cent.loc[n, "degree_weighted"] * 200
                if n in cent.index else 400
                for n in G.nodes()
            ]
        else:
            sizes = self._node_sizes_from_degree(G)

        labels = {n: self._label(n) for n in G.nodes()}
        nx.draw_networkx(
            G, pos, ax=axes[0],
            node_color=node_colors_comm, node_size=sizes,
            labels=labels, font_size=8, font_color="white",
            edge_color="gray", alpha=0.85, with_labels=True,
        )
        axes[0].set_title("Comunidades (Louvain)", fontsize=11)
        axes[0].axis("off")

        # Leyenda de comunidades
        n_com = max(self._partition.values()) + 1 if self._partition else 1
        patches = [
            mpatches.Patch(color=COMM_COLORS[i % len(COMM_COLORS)], label=f"Comunidad {i+1}")
            for i in range(min(n_com, 6))
        ]
        axes[0].legend(handles=patches, loc="lower left", fontsize=7,
                       framealpha=0.7, title="Grupos", title_fontsize=7)

        # ── Subplot derecho: betweenness ──────────────────────────────────────
        bc_col = "betweenness" if not cent.empty and "betweenness" in cent.columns else None
        if bc_col and not cent.empty:
            bc_map = cent["betweenness"].to_dict()
        else:
            bc_map = nx.betweenness_centrality(G, weight="weight", normalized=True)

        node_bc = [bc_map.get(n, 0) for n in G.nodes()]

        nc = nx.draw_networkx(
            G, pos, ax=axes[1],
            node_color=node_bc, cmap=plt.cm.YlOrRd,
            node_size=600, labels=labels,
            edge_color="gray", alpha=0.85,
            with_labels=True, font_size=7,
        )
        axes[1].set_title("Centralidad de intermediación (betweenness)", fontsize=11)
        axes[1].axis("off")

        # Colorbar manual para betweenness
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                   norm=plt.Normalize(vmin=min(node_bc), vmax=max(node_bc) or 1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[1], shrink=0.6, label="Betweenness normalizado")

        fig.tight_layout()
        self._savefig("fig2_1_red_principal.png", fig)

    def _fig_subgrafos_periodo(self) -> None:
        """
        Figura 2-2: fila de subgrafos, uno por período histórico.
        Estilo flat, color sólido por período, sin ejes.
        """
        if not _HAS_NX:
            return

        periodos = ["Pre-golpe", "23-F (1981)", "Proceso judicial", "Post-proceso"]
        subgrafos: dict[str, nx.Graph] = {}
        for periodo in periodos:
            subset = self.df[self.df["periodo"] == periodo]
            if len(subset) == 0:
                continue
            personas = self._extraer_personas(subset)
            G_p = self._build_graph(personas)
            if G_p.number_of_nodes() > 1:
                subgrafos[periodo] = self._subgrafo_top(G_p, max_nodos=30)

        if not subgrafos:
            logger.warning("Sin subgrafos temporales con datos suficientes.")
            return

        n = len(subgrafos)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        fig.suptitle("Red de actores por período histórico", fontsize=13, fontweight="bold")

        for ax, (periodo, G_t) in zip(axes, subgrafos.items()):
            pos_t = nx.spring_layout(G_t, seed=42)
            color = PERIODO_COLORS.get(periodo, "#888888")
            nx.draw_networkx(
                G_t, pos_t, ax=ax,
                node_color=color, node_size=500,
                font_size=7, font_color="white",
                edge_color="gray", alpha=0.85, with_labels=True,
                labels={n: self._label(n) for n in G_t.nodes()},
            )
            ax.set_title(
                f"{periodo}\n({G_t.number_of_nodes()} actores, {G_t.number_of_edges()} vínculos)",
                fontsize=10,
            )
            ax.axis("off")

        fig.tight_layout()
        self._savefig("fig2_2_subgrafos_periodo.png", fig)