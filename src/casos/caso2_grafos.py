"""ActorGraphCaso: grafo de co-menciones de actores entre documentos del 23-F."""

from __future__ import annotations

import ast
import html
import logging
import re
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

# ── Imports opcionales ─────────────────────────────────────────────────────────
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


# ── Paleta corporativa coherente con el resto del proyecto ─────────────────────
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
        self._period_graphs: dict[str, "nx.Graph"] = {}

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

        self._partition = self.detect_communities()
        self._centrality_df = self.compute_centrality()

        top5 = (
            self._centrality_df.nlargest(5, "pagerank")[["nodo", "pagerank"]]
            .to_string(index=False)
        )
        logger.info("Top 5 por PageRank:\n%s", top5)

        # ── Figuras ────────────────────────────────────────────────────────────
        self._fig_red_principal()       # Fig 2-1: comunidades + betweenness (2 subplots)
        self._fig_subgrafos_periodo()   # Fig 2-2: fila de subgrafos temporales
        self._html_panel_interactivo()  # Fig 2-3: red explorable en navegador

        panel_html = self.fig_dir / "fig2_3_panel_grafo_interactivo.html"

        self._results = {
            "n_nodos": self._graph.number_of_nodes(),
            "n_aristas": self._graph.number_of_edges(),
            "n_comunidades": len(set(self._partition.values())),
            "actor_pagerank_top": (
                self._centrality_df.iloc[0]["nodo"]
                if not self._centrality_df.empty else "N/A"
            ),
            "densidad": round(nx.density(self._graph), 4),
            "panel_interactivo": str(panel_html) if panel_html.exists() else "N/A",
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
        pagerank = nx.pagerank(G, weight="weight")
        G_dist = G.copy()
        for _, _, data in G_dist.edges(data=True):
            data["distance"] = 1 / max(float(data.get("weight", 1)), 1.0)
        betweenness = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)
        closeness = nx.closeness_centrality(G_dist, distance="distance")
        clustering = nx.clustering(G, weight="weight")
        degree = dict(G.degree(weight="weight"))
        records = [
            {
                "nodo":            n,
                "pagerank":        round(pagerank[n], 6),
                "betweenness":     round(betweenness[n], 6),
                "closeness":       round(closeness[n], 6),
                "clustering":      round(clustering[n], 6),
                "degree_weighted": degree[n],
                "n_docs":          G.nodes[n].get("n_docs", 0),
                "n_menciones":     G.nodes[n].get("n_menciones", 0),
                "comunidad":       self._partition.get(n, -1),
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
        if _HAS_SPACY and "texto_ocr" in df.columns:
            return self._ner_spacy(df)
        return [self._normalizar_lista_personas(row.get("personas")) for _, row in df.iterrows()]

    def _ner_spacy(self, df: pd.DataFrame) -> list[list[str]]:
        try:
            nlp = spacy.load("es_core_news_lg")
        except OSError:
            logger.warning("Modelo es_core_news_lg no encontrado. Usando fallback RTVE.")
            return [self._normalizar_lista_personas(row.get("personas")) for _, row in df.iterrows()]
        result: list[list[str]] = []
        for texto in df["texto_ocr"].fillna(""):
            doc = nlp(texto[:100_000])
            result.append(
                [
                    self._normalizar_nombre(ent.text)
                    for ent in doc.ents
                    if ent.label_ == "PER" and self._normalizar_nombre(ent.text)
                ]
            )
        return result

    def _build_graph(self, personas_por_doc: list[list[str]]) -> "nx.Graph":
        G = nx.Graph()
        docs_global = Counter(p for doc in personas_por_doc for p in set(doc))
        menciones_global = Counter(p for doc in personas_por_doc for p in doc)
        for nodo, n_docs in docs_global.items():
            G.add_node(nodo, n_docs=n_docs, n_menciones=menciones_global[nodo])
        for personas in personas_por_doc:
            for a, b in combinations(set(personas), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
        # Eliminar nodos aislados (sin aristas)
        G.remove_nodes_from([n for n, d in dict(G.degree()).items() if d == 0])
        return G

    def _normalizar_lista_personas(self, value) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                value = parsed if isinstance(parsed, list) else [value]
            except (ValueError, SyntaxError):
                value = [v.strip() for v in value.split(",")]
        personas = []
        for item in value if isinstance(value, (list, tuple, set)) else [value]:
            nombre = self._normalizar_nombre(str(item))
            if nombre:
                personas.append(nombre)
        return personas

    @staticmethod
    def _normalizar_nombre(nombre: str) -> str:
        nombre = nombre.split(":", 1)[0]
        nombre = " ".join(nombre.replace("---", " ").replace("\n", " ").split())
        nombre = re.sub(r"\b(sr|sra|d|don|doña)\.?\s+", "", nombre, flags=re.IGNORECASE)
        nombre = nombre.strip(" .;,-").lower()
        ruido = {
            "no consta", "tribunal", "defensores", "procesados", "periodistas",
            "ministerio fiscal", "familias", "comisiones militares",
            "consejo supremo de justicia militar",
        }
        if not nombre or nombre in ruido or len(nombre) < 3:
            return ""
        return nombre

    def _subgrafo_top(self, G: "nx.Graph", max_nodos: int = 60) -> "nx.Graph":
        """Recorta a los max_nodos con mayor degree para legibilidad."""
        if G.number_of_nodes() <= max_nodos:
            return G
        top = sorted(dict(G.degree(weight="weight")), key=lambda n: G.degree(n, weight="weight"), reverse=True)[:max_nodos]
        return G.subgraph(top).copy()

    def _crear_subgrafos_periodo(self, max_nodos: int = 30) -> dict[str, "nx.Graph"]:
        periodos = ["Pre-golpe", "23-F (1981)", "Proceso judicial", "Post-proceso"]
        subgrafos: dict[str, nx.Graph] = {}
        for periodo in periodos:
            subset = self.df[self.df["periodo"] == periodo]
            if len(subset) == 0:
                continue
            personas = self._extraer_personas(subset)
            G_p = self._build_graph(personas)
            if G_p.number_of_nodes() > 1:
                subgrafos[periodo] = self._subgrafo_top(G_p, max_nodos=max_nodos)
        return subgrafos

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
          Izquierda — comunidades coloreadas (nodo - degree).
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
            G_dist = G.copy()
            for _, _, data in G_dist.edges(data=True):
                data["distance"] = 1 / max(float(data.get("weight", 1)), 1.0)
            bc_map = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)

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

        subgrafos = self._crear_subgrafos_periodo(max_nodos=30)
        self._period_graphs = subgrafos

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

    def _html_panel_interactivo(self) -> None:
        """Genera un panel HTML con red explorable, resumen y tabla filtrable."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly no disponible — no se genera el panel interactivo.")
            return

        redes = {"Red completa": self._subgrafo_top(self._graph, max_nodos=75)}
        redes.update(self._period_graphs or self._crear_subgrafos_periodo(max_nodos=45))

        fig = go.Figure()
        buttons = []
        trace_names: list[str] = []
        for nombre, G in redes.items():
            edge_trace, node_trace = self._plotly_traces_red(G, nombre)
            fig.add_trace(edge_trace)
            fig.add_trace(node_trace)
            trace_names.extend([nombre, nombre])

        for nombre in redes:
            visible = [trace_name == nombre for trace_name in trace_names]
            buttons.append({
                "label": nombre,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Figura 2-3 — Red interactiva de actores: {nombre}"},
                ],
            })

        fig.update_layout(
            title="Figura 2-3 — Red interactiva de actores: Red completa",
            template="plotly_white",
            height=720,
            showlegend=False,
            margin=dict(l=20, r=20, t=90, b=20),
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        html_fig = fig.to_html(full_html=False, include_plotlyjs=True)
        panel = self._envolver_panel_html(html_fig)
        path = self.fig_dir / "fig2_3_panel_grafo_interactivo.html"
        path.write_text(panel, encoding="utf-8")
        logger.info("Panel interactivo guardado: %s", path)

    def _plotly_traces_red(self, G: "nx.Graph", nombre: str):
        import plotly.graph_objects as go

        pos = nx.spring_layout(G, seed=42, k=2.1)
        edge_x, edge_y = [], []
        for a, b, data in G.edges(data=True):
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.7, color="rgba(80, 86, 96, 0.35)"),
            hoverinfo="skip",
            visible=(nombre == "Red completa"),
        )

        cent = self._centrality_df.set_index("nodo") if not self._centrality_df.empty else pd.DataFrame()
        degrees = dict(G.degree(weight="weight"))
        max_degree = max(degrees.values()) if degrees else 1
        node_x, node_y, sizes, colors, texts, custom = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            row = cent.loc[node] if not cent.empty and node in cent.index else {}
            comunidad = int(self._partition.get(node, -1))
            node_x.append(x)
            node_y.append(y)
            sizes.append(12 + 34 * degrees.get(node, 1) / max_degree)
            colors.append(COMM_COLORS[comunidad % len(COMM_COLORS)] if comunidad >= 0 else "#888888")
            texts.append(self._label(node))
            custom.append([
                node,
                int(G.nodes[node].get("n_docs", 0)),
                float(row.get("pagerank", 0)) if hasattr(row, "get") else 0,
                float(row.get("betweenness", 0)) if hasattr(row, "get") else 0,
                int(degrees.get(node, 0)),
                comunidad + 1 if comunidad >= 0 else "N/A",
            ])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=texts,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="#ffffff"), opacity=0.92),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Documentos: %{customdata[1]}<br>"
                "PageRank: %{customdata[2]:.4f}<br>"
                "Betweenness: %{customdata[3]:.4f}<br>"
                "Degree ponderado: %{customdata[4]}<br>"
                "Comunidad: %{customdata[5]}<extra></extra>"
            ),
            visible=(nombre == "Red completa"),
        )
        return edge_trace, node_trace

    def _tabla_top_actores_html(self) -> str:
        if self._centrality_df.empty:
            return ""
        cols = [
            "nodo", "pagerank", "betweenness", "closeness",
            "degree_weighted", "n_docs", "n_menciones", "comunidad",
        ]
        df_top = self._centrality_df[cols].head(40).copy()
        df_top["nodo"] = df_top["nodo"].map(lambda x: html.escape(str(x).title()))
        df_top["comunidad"] = df_top["comunidad"].map(lambda x: int(x) + 1 if int(x) >= 0 else "N/A")
        return df_top.to_html(index=False, classes="tabla", border=0, table_id="actores")

    def _tabla_resumen_periodos_html(self) -> str:
        filas = []
        for periodo, G in (self._period_graphs or self._crear_subgrafos_periodo(max_nodos=45)).items():
            top_actor = "N/A"
            if G.number_of_nodes():
                top_actor = self._label(max(G.nodes(), key=lambda n: G.degree(n, weight="weight")))
            filas.append({
                "periodo": periodo,
                "actores": G.number_of_nodes(),
                "vinculos": G.number_of_edges(),
                "densidad": round(nx.density(G), 3) if G.number_of_nodes() > 1 else 0,
                "actor_mas_conectado": top_actor,
            })
        return pd.DataFrame(filas).to_html(index=False, classes="tabla compacta", border=0)

    def _envolver_panel_html(self, grafico: str) -> str:
        resumen = self._tabla_resumen_periodos_html()
        actores = self._tabla_top_actores_html()
        return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panel interactivo caso 2</title>
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
    .compacta {{ max-width: 760px; }}
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
      <h1>Panel interactivo de actores del 23-F</h1>
      {grafico}
    </section>
    <section>
      <h2>Resumen por período</h2>
      {resumen}
    </section>
    <section>
      <h2>Top actores por PageRank</h2>
      <input id="filtro" type="search" placeholder="Filtrar por actor, métrica o comunidad">
      <div class="table-wrap">{actores}</div>
    </section>
  </main>
  <script>
    const filtro = document.getElementById("filtro");
    const tabla = document.getElementById("actores");
    if (filtro && tabla) {{
      const filas = Array.from(tabla.querySelectorAll("tbody tr"));
      filtro.addEventListener("input", () => {{
        const q = filtro.value.toLowerCase();
        filas.forEach((fila) => {{
          fila.style.display = fila.innerText.toLowerCase().includes(q) ? "" : "none";
        }});
      }});
    }}
  </script>
</body>
</html>"""
