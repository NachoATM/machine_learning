"""SpatioTemporalCaso: análisis espacio-temporal y detección de anomalías."""

from __future__ import annotations

import ast
import logging
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.base import BaseCaso
from src.viz.plotter import Plotter

logger = logging.getLogger(__name__)

# ── Coordenadas de referencia (diccionario hardcoded) ──────────────────────────
_COORDS: dict[str, tuple[float, float]] = {
    "madrid": (40.4168, -3.7038),
    "barcelona": (41.3851, 2.1734),
    "valencia": (39.4699, -0.3763),
    "zaragoza": (41.6488, -0.8891),
    "sevilla": (37.3891, -5.9845),
    "bilbao": (43.2630, -2.9350),
    "murcia": (37.9922, -1.1307),
    "toledo": (39.8628, -4.0273),
    "burgos": (42.3440, -3.6969),
    "valladolid": (41.6523, -4.7245),
    "salamanca": (40.9701, -5.6635),
    "alicante": (38.3452, -0.4810),
    "pamplona": (42.8125, -1.6458),
    "vitoria": (42.8467, -2.6727),
    "san sebastian": (43.3183, -1.9812),
    "la coruna": (43.3623, -8.4115),
    "granada": (37.1773, -3.5986),
    "malaga": (36.7213, -4.4213),
    "cadiz": (36.5271, -6.2886),
    "congreso de los diputados": (40.4165, -3.6972),
    "congreso": (40.4165, -3.6972),
    "palacio de la zarzuela": (40.4532, -3.7874),
    "palacio de la moncloa": (40.4386, -3.7152),
    "moncloa": (40.4386, -3.7152),
    "palace": (40.4168, -3.7038),
    "washington": (38.9072, -77.0369),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "bonn": (50.7374, 7.0982),
    "lisboa": (38.7169, -9.1395),
    "roma": (41.9028, 12.4964),
}


class SpatioTemporalCaso(BaseCaso):
    """
    Caso 5 — Análisis Espacio-Temporal y Detección de Anomalías.

    - Serie temporal multivariante de métricas del corpus por año
    - Detección de anomalías con IsolationForest
    - Mapa estático e interactivo de menciones geográficas
    - Heatmap lugar × período
    """

    _FEATURES_ANOMALIA = [
        "paginas", "n_palabras_ocr", "n_personas", "n_lugares",
        "n_keywords", "riqueza_lexica", "densidad_entidades", "compresion_resumen",
    ]

    def __init__(self, df: pd.DataFrame, output_dir="outputs", fig_dir="figuras") -> None:
        super().__init__(df, output_dir, fig_dir)
        self.plotter = Plotter(fig_dir)
        self._df_anomalias: pd.DataFrame = pd.DataFrame()
        self._df_geo: pd.DataFrame = pd.DataFrame()

    # ── API pública ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        logger.info("=== Caso 5: Análisis Espacio-Temporal ===")
        self._fig_serie_temporal()             # Fig 5-1
        self._df_anomalias = self._detectar_anomalias()
        self._fig_scatter_anomalias()          # Fig 5-2
        self._df_geo = self._geocodificar_lugares()
        self._fig_mapa_geografico()            # Fig 5-3 estática
        self._html_mapa_interactivo()          # Fig 5-3 interactiva
        self._fig_heatmap_lugar_periodo()      # Fig 5-4

        n_anom = (self._df_anomalias["anomalia"] == -1).sum() \
            if "anomalia" in self._df_anomalias.columns else 0
        mapa_html = self.fig_dir / "fig5_3_mapa_geografico_interactivo.html"

        self._results = {
            "n_anomalias": int(n_anom),
            "n_lugares_geocodificados": len(self._df_geo),
            "lugar_mas_frecuente": (
                self._df_geo.sort_values("frecuencia", ascending=False).iloc[0]["lugar"]
                if not self._df_geo.empty else "N/A"
            ),
            "mapa_interactivo": str(mapa_html) if mapa_html.exists() else "N/A",
        }
        return self._results

    def export(self) -> None:
        if not self._df_anomalias.empty:
            cols = ["id", "titulo", "fuente", "periodo", "anomalia", "anomalia_score"]
            self._save_csv("anomalias_caso5.csv",
                           self._df_anomalias[[c for c in cols if c in self._df_anomalias.columns]])
        if not self._df_geo.empty:
            self._save_csv("frecuencia_lugares.csv", self._df_geo)

    # ── Serie temporal ─────────────────────────────────────────────────────────

    def _fig_serie_temporal(self) -> None:
        df_t = self.df[self.df["anio"].notna()].copy()
        df_t["anio"] = df_t["anio"].astype(int)
        metricas = {
            "densidad_entidades": "Densidad de Entidades",
            "riqueza_lexica": "Riqueza Léxica (TTR)",
            "n_palabras_ocr": "N Palabras OCR",
        }
        metricas_presentes = {k: v for k, v in metricas.items() if k in df_t.columns}
        if not metricas_presentes:
            return

        agg = df_t.groupby("anio")[list(metricas_presentes.keys())].mean()
        n_met = len(metricas_presentes)
        fig, axes = plt.subplots(n_met, 1, figsize=(13, 3.5 * n_met), sharex=True)
        if n_met == 1:
            axes = [axes]
        fig.suptitle("Figura 5-1 — Serie Temporal de Métricas del Corpus",
                     fontweight="bold", fontsize=13)

        colores_linea = ["#d73027", "#2166ac", "#1a9850"]
        eventos = {1981: "23-F (golpe)", 1982: "Inicio juicio", 1983: "Sentencia"}

        for ax, (col, label), color in zip(axes, metricas_presentes.items(), colores_linea):
            if col not in agg.columns:
                continue
            serie = agg[col].dropna()
            ax.plot(serie.index, serie.values, "o-", color=color, linewidth=2,
                    markersize=6, label=label)
            ax.fill_between(serie.index, serie.values, alpha=0.15, color=color)
            for anio, texto in eventos.items():
                if agg.index.min() <= anio <= agg.index.max():
                    ax.axvline(anio, color="grey", linestyle=":", linewidth=1.2)
                    ax.text(anio + 0.05, ax.get_ylim()[1] * 0.92, texto,
                            fontsize=7, color="grey", rotation=90, va="top")
            ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        axes[-1].set_xlabel("Año del documento")
        fig.tight_layout()
        self._savefig("fig5_1_serie_temporal.png", fig)

    # ── Detección de anomalías ─────────────────────────────────────────────────

    def _detectar_anomalias(self) -> pd.DataFrame:
        feats = [c for c in self._FEATURES_ANOMALIA if c in self.df.columns]
        df_a = self.df[feats + ["id", "titulo", "fuente", "periodo"]].copy()
        df_a = df_a.dropna(subset=feats).reset_index(drop=True)
        if len(df_a) < 5:
            return pd.DataFrame()

        X = StandardScaler().fit_transform(df_a[feats].fillna(0))
        iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
        df_a["anomalia"] = iso.fit_predict(X)
        df_a["anomalia_score"] = iso.score_samples(X).round(4)

        n_anom = (df_a["anomalia"] == -1).sum()
        logger.info("IsolationForest: %d anomalías detectadas (de %d)", n_anom, len(df_a))
        return df_a

    def _fig_scatter_anomalias(self) -> None:
        if self._df_anomalias.empty:
            return
        feats = [c for c in self._FEATURES_ANOMALIA if c in self._df_anomalias.columns]
        X = StandardScaler().fit_transform(self._df_anomalias[feats].fillna(0))
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)

        normales = self._df_anomalias["anomalia"] == 1
        anom = self._df_anomalias["anomalia"] == -1

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(X_2d[normales, 0], X_2d[normales, 1],
                   c="#2166ac", alpha=0.6, s=50, label="Normal",
                   edgecolors="white", linewidths=0.3)
        ax.scatter(X_2d[anom, 0], X_2d[anom, 1],
                   c="#d73027", alpha=0.9, s=90, label="Anomalía",
                   edgecolors="black", linewidths=0.6, marker="D")

        # Etiquetar top anomalías
        top_anom_idx = self._df_anomalias[anom].nsmallest(8, "anomalia_score").index
        for idx in top_anom_idx:
            i = self._df_anomalias.index.get_loc(idx)
            titulo_corto = self._df_anomalias.loc[idx, "titulo"][:35] + "…"
            ax.annotate(titulo_corto, (X_2d[i, 0], X_2d[i, 1]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_title("Figura 5-2 — Detección de Anomalías (IsolationForest + PCA)",
                     fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=10)
        fig.tight_layout()
        self._savefig("fig5_2_anomalias_pca.png", fig)

    # ── Análisis geográfico ────────────────────────────────────────────────────

    @staticmethod
    def _iter_lugares(valor) -> list[str]:
        """Devuelve lugares limpios aceptando listas reales o listas serializadas."""
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return []
        if isinstance(valor, (list, tuple, set)):
            return [str(v).strip() for v in valor if str(v).strip()]
        if isinstance(valor, str):
            texto = valor.strip()
            if not texto:
                return []
            try:
                parsed = ast.literal_eval(texto)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except (SyntaxError, ValueError):
                pass
            return [texto]
        return [str(valor).strip()] if str(valor).strip() else []

    @staticmethod
    def _match_lugar(lugar: str, candidatos: set[str] | None = None) -> str | None:
        lugar_norm = SpatioTemporalCaso._normalizar_lugar(lugar)
        if not lugar_norm or lugar_norm in {"sala", "radio", "canal"}:
            return None
        keys = candidatos or set(_COORDS)
        if "estados unidos" in lugar_norm and "congreso" in lugar_norm:
            return None

        for key in sorted(keys, key=len, reverse=True):
            key_norm = SpatioTemporalCaso._normalizar_lugar(key)
            if key_norm == lugar_norm:
                return key
            key_tokens = set(key_norm.split())
            lugar_tokens = set(lugar_norm.split())
            if key_tokens and key_tokens.issubset(lugar_tokens):
                return key
            if len(lugar_norm) > 4 and lugar_tokens and lugar_tokens.issubset(key_tokens):
                return key
        return None

    @staticmethod
    def _normalizar_lugar(valor: str) -> str:
        texto = unicodedata.normalize("NFKD", str(valor).lower())
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        texto = re.sub(r"[^a-z0-9]+", " ", texto)
        return re.sub(r"\s+", " ", texto).strip()

    @staticmethod
    def _separar_puntos_superpuestos(df_geo: pd.DataFrame) -> pd.DataFrame:
        """Añade coordenadas de visualización para lugares con puntos solapados."""
        if df_geo.empty:
            return df_geo
        df_geo = df_geo.copy()
        df_geo["lat_plot"] = df_geo["lat"]
        df_geo["lon_plot"] = df_geo["lon"]
        grupos = df_geo.groupby([df_geo["lat"].round(3), df_geo["lon"].round(3)]).groups
        for indices in grupos.values():
            indices = list(indices)
            if len(indices) == 1:
                continue
            radio = 0.055
            for i, idx in enumerate(indices):
                angulo = 2 * np.pi * i / len(indices)
                df_geo.loc[idx, "lat_plot"] = df_geo.loc[idx, "lat"] + np.sin(angulo) * radio
                df_geo.loc[idx, "lon_plot"] = df_geo.loc[idx, "lon"] + np.cos(angulo) * radio
        return df_geo

    def _geocodificar_lugares(self) -> pd.DataFrame:
        from collections import Counter

        todos = [
            lugar.lower()
            for lugares_doc in self.df["lugares"].apply(self._iter_lugares)
            for lugar in lugares_doc
        ]
        freq = Counter(todos)
        docs_por_lugar = {key: 0 for key in _COORDS}
        periodos_por_lugar: dict[str, Counter] = {key: Counter() for key in _COORDS}

        for _, row in self.df.iterrows():
            vistos_doc = set()
            for lugar in self._iter_lugares(row.get("lugares")):
                key = self._match_lugar(lugar)
                if key:
                    vistos_doc.add(key)
                    periodos_por_lugar[key][row.get("periodo", "Desconocido")] += 1
            for key in vistos_doc:
                docs_por_lugar[key] += 1

        geo_agg: dict[str, dict] = {}
        raw_por_lugar: dict[str, Counter] = {key: Counter() for key in _COORDS}
        for lugar, frec in freq.most_common():
            lugar_norm = self._match_lugar(lugar)
            if lugar_norm:
                lat, lon = _COORDS[lugar_norm]
                raw_por_lugar[lugar_norm][lugar] += frec
                if lugar_norm not in geo_agg:
                    geo_agg[lugar_norm] = {
                        "lugar": lugar_norm,
                        "frecuencia": 0,
                        "n_documentos": docs_por_lugar.get(lugar_norm, 0),
                        "periodo_principal": (
                            periodos_por_lugar[lugar_norm].most_common(1)[0][0]
                            if periodos_por_lugar[lugar_norm] else "Desconocido"
                        ),
                        "lat": lat,
                        "lon": lon,
                    }
                geo_agg[lugar_norm]["frecuencia"] += frec

        registros = []
        for lugar_norm, registro in geo_agg.items():
            registro = registro.copy()
            registro["lugar_raw"] = "; ".join(
                raw for raw, _ in raw_por_lugar[lugar_norm].most_common(4)
            )
            registros.append(registro)

        df_geo = (
            pd.DataFrame(registros)
            .sort_values(["frecuencia", "n_documentos"], ascending=False)
            .reset_index(drop=True)
        )
        df_geo = self._separar_puntos_superpuestos(df_geo)
        logger.info("Lugares geocodificados: %d de %d únicos", len(df_geo), len(freq))
        return df_geo

    def _fig_mapa_geografico(self) -> None:
        if self._df_geo.empty:
            return
        fig, ax = plt.subplots(figsize=(13, 8))

        max_frec = self._df_geo["frecuencia"].max()
        sizes = 90 + np.sqrt(self._df_geo["frecuencia"] / max_frec) * 850

        scatter = ax.scatter(
            self._df_geo["lon_plot"], self._df_geo["lat_plot"],
            s=sizes, c=self._df_geo["frecuencia"],
            cmap="YlOrRd", alpha=0.72, edgecolors="#222", linewidths=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="N menciones")

        for _, row in self._df_geo.nlargest(12, "frecuencia").iterrows():
            ax.annotate(
                f"{row['lugar'].title()} ({row['frecuencia']})",
                (row["lon_plot"], row["lat_plot"]),
                fontsize=8, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
            )

        lons = self._df_geo["lon_plot"]
        lats = self._df_geo["lat_plot"]
        ax.set_xlim(lons.min() - 2, lons.max() + 2)
        ax.set_ylim(lats.min() - 1.5, lats.max() + 1.5)

        ax.set_title("Figura 5-3 — Mapa de Menciones Geográficas en el Corpus",
                     fontweight="bold")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._savefig("fig5_3_mapa_geografico.png", fig)

    def _html_mapa_interactivo(self) -> None:
        if self._df_geo.empty:
            return
        try:
            import plotly.express as px
        except ImportError:
            logger.warning("plotly no disponible — no se genera el mapa interactivo.")
            return

        df_plot = self._df_geo.copy()
        df_plot["lugar_label"] = df_plot["lugar"].str.title()
        df_plot["tamano"] = np.sqrt(df_plot["frecuencia"]).clip(lower=1)

        fig = px.scatter_geo(
            df_plot,
            lat="lat_plot",
            lon="lon_plot",
            size="tamano",
            color="frecuencia",
            hover_name="lugar_label",
            hover_data={
                "lat": ":.4f",
                "lon": ":.4f",
                "lat_plot": False,
                "lon_plot": False,
                "frecuencia": True,
                "n_documentos": True,
                "periodo_principal": True,
                "tamano": False,
                "lugar_label": False,
            },
            text="lugar_label",
            color_continuous_scale="YlOrRd",
            size_max=34,
            projection="natural earth",
            title="Figura 5-3 — Mapa Interactivo de Menciones Geográficas",
        )
        fig.update_traces(
            mode="markers+text",
            marker=dict(line=dict(width=1, color="#2b2b2b"), opacity=0.78),
            textposition="top center",
            textfont=dict(size=10, color="#222"),
        )
        fig.update_geos(
            fitbounds="locations",
            visible=True,
            showland=True,
            landcolor="#f7f4eb",
            showcountries=True,
            countrycolor="#9aa0a6",
            showocean=True,
            oceancolor="#dceaf5",
            lataxis_showgrid=True,
            lonaxis_showgrid=True,
        )
        fig.update_layout(
            width=1100,
            height=720,
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_colorbar=dict(title="Menciones"),
            font=dict(family="Arial", size=13),
        )

        path = self.fig_dir / "fig5_3_mapa_geografico_interactivo.html"
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        logger.info("Mapa interactivo guardado: %s", path)

    def _fig_heatmap_lugar_periodo(self) -> None:
        if self._df_geo.empty:
            return
        lugares_top = set(self._df_geo.nlargest(20, "frecuencia")["lugar"])
        periodos_orden = ["Pre-golpe", "23-F (1981)", "Proceso judicial", "Post-proceso"]

        # Construir tabla: lugar × período
        matriz = pd.DataFrame(0, index=sorted(lugares_top), columns=periodos_orden)
        for _, row in self.df.iterrows():
            if row["periodo"] not in periodos_orden:
                continue
            for lugar in self._iter_lugares(row.get("lugares")):
                key = self._match_lugar(lugar, lugares_top)
                if key:
                    matriz.loc[key, row["periodo"]] += 1

        # Eliminar filas vacías
        matriz = matriz[matriz.sum(axis=1) > 0]
        if matriz.empty:
            return

        fig, ax = plt.subplots(figsize=(11, max(5, len(matriz) * 0.4)))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.3, linecolor="white", ax=ax,
                    cbar_kws={"label": "N menciones"})
        ax.set_yticklabels([n.title() for n in matriz.index], rotation=0, fontsize=9)
        ax.set_xticklabels(periodos_orden, rotation=15)
        ax.set_title("Figura 5-4 — Menciones Geográficas × Período Histórico",
                     fontweight="bold")
        ax.set_ylabel("Lugar")
        ax.set_xlabel("Período")
        fig.tight_layout()
        self._savefig("fig5_4_heatmap_lugar_periodo.png", fig)
