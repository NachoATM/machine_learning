"""ClasificadorCaso: clasificador supervisado de ministerio de origen."""

from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.base import BaseCaso
from src.viz.plotter import Plotter

logger = logging.getLogger(__name__)


class ClasificadorCaso(BaseCaso):
    """
    Caso 4 — Clasificador Supervisado de Tipo Documental.

    Target: ministerio de origen (Defensa / Interior / Exteriores).
    Features: TF-IDF (ngram 1-2) + variables numéricas en un Pipeline.
    Compara LogisticRegression, RandomForest y LinearSVC con CV estratificado.
    Las métricas, matriz de confusión y errores usan predicción out-of-fold para
    evitar estimaciones optimistas.
    """

    _MODELOS: dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", C=1.0
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1
        ),
        "LinearSVC": LinearSVC(
            max_iter=2000, random_state=42, class_weight="balanced", C=1.0
        ),
    }

    _FEATURES_NUM = [
        "paginas", "n_palabras_ocr", "n_personas", "n_lugares",
        "n_keywords", "riqueza_lexica", "densidad_entidades", "compresion_resumen",
    ]

    def __init__(self, df: pd.DataFrame, output_dir="outputs", fig_dir="figuras") -> None:
        super().__init__(df, output_dir, fig_dir)
        self.plotter = Plotter(fig_dir)
        self._mejor_modelo_nombre: str = ""
        self._mejor_modelo = None
        self._le = LabelEncoder()
        self._df_train: pd.DataFrame = pd.DataFrame()
        self._resultados_cv: dict[str, dict[str, float]] = {}
        self._errores: pd.DataFrame = pd.DataFrame()
        self._features_num_usadas: list[str] = []
        self._reporte_oof: dict = {}
        self._y_pred = np.array([])
        self._y_encoded = np.array([])

    # ── API pública ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        logger.info("=== Caso 4: Clasificador Supervisado ===")

        df_c = self._preparar_datos()
        if df_c is None or len(df_c) < 10:
            logger.error("Datos insuficientes para clasificación.")
            self._results = {"error": "datos insuficientes"}
            return self._results

        X, y = self._build_features(df_c)
        self._df_train = df_c
        self._y_encoded = y

        cv = self._crear_cv(y)
        self._resultados_cv = self._evaluar_modelos(X, y, cv)
        self._mejor_modelo_nombre = max(
            self._resultados_cv,
            key=lambda nombre: self._resultados_cv[nombre]["f1_macro_mean"],
        )
        logger.info("Mejor modelo: %s (F1 macro CV=%.3f)",
                    self._mejor_modelo_nombre,
                    self._resultados_cv[self._mejor_modelo_nombre]["f1_macro_mean"])

        self._mejor_modelo = self._crear_pipeline(self._MODELOS[self._mejor_modelo_nombre])
        y_pred = cross_val_predict(self._mejor_modelo, X, y, cv=cv, n_jobs=1)
        self._mejor_modelo.fit(X, y)
        self._y_pred = y_pred
        self._X = X
        self._reporte_oof = classification_report(
            y,
            y_pred,
            target_names=self._le.classes_,
            output_dict=True,
            zero_division=0,
        )

        self._errores = self._analizar_errores(df_c, y, y_pred)

        self._html_panel_interactivo(y, y_pred)  # Fig 4-4
        self._fig_comparativa_f1()             # Fig 4-1
        self._fig_confusion_matrix(y, y_pred)  # Fig 4-2
        self._fig_feature_importance()         # Fig 4-3

        panel_html = self.fig_dir / "fig4_4_panel_clasificador_interactivo.html"
        self._results = {
            "mejor_modelo": self._mejor_modelo_nombre,
            "f1_macro_cv": round(
                self._resultados_cv[self._mejor_modelo_nombre]["f1_macro_mean"], 4
            ),
            "f1_macro_oof": round(self._reporte_oof["macro avg"]["f1-score"], 4),
            "accuracy_oof": round(self._reporte_oof["accuracy"], 4),
            "clases": list(self._le.classes_),
            "n_train": len(df_c),
            "n_errores": len(self._errores),
            "panel_interactivo": str(panel_html) if panel_html.exists() else "N/A",
        }
        return self._results

    def export(self) -> None:
        if not self._df_train.empty:
            df_pred = self._df_train[["id", "titulo", "fuente"]].copy()
            if len(self._y_pred) == len(df_pred):
                df_pred["fuente_pred"] = self._le.inverse_transform(self._y_pred)
                df_pred["correcto"] = df_pred["fuente"] == df_pred["fuente_pred"]
                df_pred["tipo_prediccion"] = "out_of_fold"
            self._save_csv("predicciones_caso4.csv", df_pred)
        if self._mejor_modelo is not None:
            path = self.output_dir / "modelo_caso4.joblib"
            joblib.dump(
                {
                    "pipeline": self._mejor_modelo,
                    "label_encoder": self._le,
                    "features_num": self._features_num_usadas,
                    "clases": list(self._le.classes_),
                },
                path,
            )
            logger.info("Modelo guardado: %s", path)
        if not self._errores.empty:
            self._save_csv("errores_caso4.csv", self._errores)

    # ── Preparación de datos ───────────────────────────────────────────────────

    def _preparar_datos(self) -> pd.DataFrame | None:
        """Filtra solo los ministerios con suficientes muestras."""
        df = self.df[self.df["fuente"].isin(["Defensa", "Interior", "Exteriores"])].copy()
        df = df[df["n_palabras_ocr"] >= 20]
        conteo = df["fuente"].value_counts()
        clases_validas = conteo[conteo >= 3].index.tolist()
        df = df[df["fuente"].isin(clases_validas)].reset_index(drop=True)
        logger.info("Distribución de clases:\n%s", df["fuente"].value_counts().to_string())
        return df if len(df) >= 10 else None

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepara columnas de entrada; el fit real vive dentro del Pipeline."""
        X = df.copy()
        X["texto_limpio"] = X["texto_ocr"].fillna("").apply(self.plotter.limpiar_texto)
        self._features_num_usadas = [c for c in self._FEATURES_NUM if c in X.columns]
        y = self._le.fit_transform(df["fuente"])
        return X, y

    def _crear_preprocesador(self) -> ColumnTransformer:
        transformers = [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.92,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                "texto_limpio",
            )
        ]
        if self._features_num_usadas:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self._features_num_usadas,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _crear_pipeline(self, estimador) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", self._crear_preprocesador()),
                ("model", clone(estimador)),
            ]
        )

    # ── Evaluación ─────────────────────────────────────────────────────────────

    @staticmethod
    def _crear_cv(y) -> StratifiedKFold:
        n_splits = max(2, min(5, int(np.bincount(y).min())))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _evaluar_modelos(self, X, y, cv) -> dict[str, dict[str, float]]:
        resultados: dict[str, dict[str, float]] = {}
        for nombre, modelo in self._MODELOS.items():
            pipe = self._crear_pipeline(modelo)
            scores = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=("f1_macro", "accuracy"),
                n_jobs=1,
                error_score="raise",
            )
            resultados[nombre] = {
                "f1_macro_mean": float(scores["test_f1_macro"].mean()),
                "f1_macro_std": float(scores["test_f1_macro"].std()),
                "accuracy_mean": float(scores["test_accuracy"].mean()),
            }
            logger.info(
                "%s — F1 macro CV: %.3f ± %.3f | Accuracy: %.3f",
                nombre,
                resultados[nombre]["f1_macro_mean"],
                resultados[nombre]["f1_macro_std"],
                resultados[nombre]["accuracy_mean"],
            )
        return resultados

    def _analizar_errores(self, df: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
        mask_error = y_true != y_pred
        if not mask_error.any():
            return pd.DataFrame()
        df_err = df[mask_error].copy()
        df_err["real"] = self._le.inverse_transform(y_true[mask_error])
        df_err["prediccion"] = self._le.inverse_transform(y_pred[mask_error])
        cols = ["id", "titulo", "fuente", "real", "prediccion"]
        return df_err[cols].reset_index(drop=True)

    # ── Figuras ────────────────────────────────────────────────────────────────

    def _fig_comparativa_f1(self) -> None:
        nombres = list(self._resultados_cv.keys())
        scores = [self._resultados_cv[n]["f1_macro_mean"] for n in nombres]
        stds = [self._resultados_cv[n]["f1_macro_std"] for n in nombres]
        colores = ["#2166ac" if n == self._mejor_modelo_nombre else "#aaaaaa"
                   for n in nombres]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            nombres,
            scores,
            yerr=stds,
            capsize=4,
            color=colores,
            edgecolor="white",
            width=0.5,
        )
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0, min(1.1, max(scores) + 0.12))
        ax.set_ylabel("F1 Macro (validación cruzada)")
        ax.set_title("Figura 4-1 — Comparativa de Modelos (F1 Macro)", fontweight="bold")
        ax.axhline(max(scores), color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        self._savefig("fig4_1_comparativa_modelos.png", fig)

    def _fig_confusion_matrix(self, y_true, y_pred) -> None:
        clases = self._le.classes_
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Figura 4-2 — Matriz de Confusión OOF — {self._mejor_modelo_nombre}",
                     fontweight="bold")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=clases, yticklabels=clases, ax=axes[0],
                    linewidths=0.5, linecolor="white")
        axes[0].set_title("A — Conteos absolutos")
        axes[0].set_xlabel("Predicción")
        axes[0].set_ylabel("Real")
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=clases, yticklabels=clases, ax=axes[1],
                    linewidths=0.5, linecolor="white", vmin=0, vmax=1)
        axes[1].set_title("B — Normalizada por fila")
        axes[1].set_xlabel("Predicción")
        axes[1].set_ylabel("Real")
        fig.tight_layout()
        self._savefig("fig4_2_confusion_matrix.png", fig)

    def _fig_feature_importance(self) -> None:
        clases = self._le.classes_
        if self._mejor_modelo is None:
            return

        preprocess = self._mejor_modelo.named_steps["preprocess"]
        modelo_final = self._mejor_modelo.named_steps["model"]
        tfidf = preprocess.named_transformers_["tfidf"]
        feat_names_tfidf = [f"txt: {name}" for name in tfidf.get_feature_names_out()]
        feat_names_num = [f"num: {name}" for name in self._features_num_usadas]
        all_feats = feat_names_tfidf + feat_names_num

        if hasattr(modelo_final, "coef_"):
            # LogisticRegression / LinearSVC → coeficientes por clase
            coef = modelo_final.coef_  # (n_clases, n_features) o (1, n_features)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            n_top = 15
            fig, axes = plt.subplots(1, len(clases), figsize=(5 * len(clases), 6))
            if len(clases) == 1:
                axes = [axes]
            fig.suptitle("Figura 4-3 — Top Features Discriminantes por Clase",
                         fontweight="bold")
            for ax, i, clase in zip(axes, range(len(clases)), clases):
                c = coef[i] if i < len(coef) else coef[0]
                top_pos = np.argsort(c)[::-1][:n_top]
                top_neg = np.argsort(c)[:n_top]
                idx = np.concatenate([top_pos, top_neg])
                idx = idx[:n_top]
                vals = c[idx]
                names = [all_feats[j] if j < len(all_feats) else f"feat_{j}" for j in idx]
                colores = ["#d73027" if v > 0 else "#2166ac" for v in vals]
                ax.barh(range(len(names)), vals, color=colores, alpha=0.8)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_title(clase, fontweight="bold")
            fig.tight_layout()

        elif hasattr(modelo_final, "feature_importances_"):
            # RandomForest
            importances = modelo_final.feature_importances_
            top_idx = importances.argsort()[::-1][:20]
            names = [all_feats[i] if i < len(all_feats) else f"feat_{i}" for i in top_idx]
            vals = importances[top_idx]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(names)), vals, color="#2166ac", alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.set_title("Figura 4-3 — Feature Importance (RandomForest) — Top 20",
                         fontweight="bold")
            ax.set_xlabel("Importance")
            fig.tight_layout()
        else:
            return

        self._savefig("fig4_3_feature_importance.png", fig)

    def _html_panel_interactivo(self, y_true, y_pred) -> None:
        """Genera un panel HTML explorable con métricas y errores del clasificador."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly no disponible — no se genera el panel interactivo.")
            return

        clases = list(self._le.classes_)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        nombres = list(self._resultados_cv.keys())
        f1_scores = [self._resultados_cv[n]["f1_macro_mean"] for n in nombres]
        f1_stds = [self._resultados_cv[n]["f1_macro_std"] for n in nombres]
        colores = ["#2166ac" if n == self._mejor_modelo_nombre else "#9aa0a6"
                   for n in nombres]

        clases_reporte = [c for c in clases if c in self._reporte_oof]
        f1_clase = [self._reporte_oof[c]["f1-score"] for c in clases_reporte]
        soporte_clase = [self._reporte_oof[c]["support"] for c in clases_reporte]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Comparativa de modelos",
                "Matriz de confusión normalizada",
                "F1 por clase",
                "Rasgos más influyentes del modelo final",
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        fig.add_trace(
            go.Bar(
                x=nombres,
                y=f1_scores,
                error_y=dict(type="data", array=f1_stds, visible=True),
                marker_color=colores,
                customdata=[
                    [self._resultados_cv[n]["accuracy_mean"], self._resultados_cv[n]["f1_macro_std"]]
                    for n in nombres
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>F1 macro: %{y:.3f}"
                    "<br>Std: %{customdata[1]:.3f}"
                    "<br>Accuracy: %{customdata[0]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=clases,
                y=clases,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                text=cm,
                texttemplate="%{text}<br>%{z:.2f}",
                hovertemplate=(
                    "Real: %{y}<br>Predicción: %{x}"
                    "<br>Casos: %{text}<br>Proporción: %{z:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=clases_reporte,
                y=f1_clase,
                marker_color="#1a9850",
                customdata=soporte_clase,
                hovertemplate="<b>%{x}</b><br>F1: %{y:.3f}<br>Soporte: %{customdata}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        top_features = self._top_features_interactivas(n=18)
        if not top_features.empty:
            fig.add_trace(
                go.Bar(
                    x=top_features["peso"],
                    y=top_features["feature"],
                    orientation="h",
                    marker_color=top_features["color"],
                    customdata=top_features["detalle"],
                    hovertemplate="<b>%{y}</b><br>Peso: %{x:.4f}<br>%{customdata}<extra></extra>",
                ),
                row=2,
                col=2,
            )

        fig.update_yaxes(autorange="reversed", row=2, col=2)
        fig.update_yaxes(title_text="Real", row=1, col=2)
        fig.update_xaxes(title_text="Predicción", row=1, col=2)
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_layout(
            title=f"Figura 4-4 — Panel Interactivo del Clasificador ({self._mejor_modelo_nombre})",
            height=900,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=60, r=40, t=90, b=50),
        )

        pred_table = self._tabla_predicciones_html()
        metricas_table = self._tabla_metricas_html()
        html_fig = fig.to_html(full_html=False, include_plotlyjs=True)
        contenido = self._envolver_panel_html(html_fig, metricas_table, pred_table)
        path = self.fig_dir / "fig4_4_panel_clasificador_interactivo.html"
        path.write_text(contenido, encoding="utf-8")
        logger.info("Panel interactivo guardado: %s", path)

    def _top_features_interactivas(self, n: int = 18) -> pd.DataFrame:
        if self._mejor_modelo is None:
            return pd.DataFrame()

        preprocess = self._mejor_modelo.named_steps["preprocess"]
        modelo_final = self._mejor_modelo.named_steps["model"]
        tfidf = preprocess.named_transformers_["tfidf"]
        all_feats = (
            [f"txt: {name}" for name in tfidf.get_feature_names_out()]
            + [f"num: {name}" for name in self._features_num_usadas]
        )

        if hasattr(modelo_final, "coef_"):
            coef = modelo_final.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            filas = []
            for i, clase in enumerate(self._le.classes_):
                pesos = coef[i] if i < len(coef) else coef[0]
                idx = np.argsort(np.abs(pesos))[::-1][:max(1, n // len(self._le.classes_))]
                for j in idx:
                    filas.append({
                        "feature": all_feats[j] if j < len(all_feats) else f"feat_{j}",
                        "peso": float(pesos[j]),
                        "detalle": f"Clase: {clase}",
                        "color": "#d73027" if pesos[j] >= 0 else "#2166ac",
                    })
            return pd.DataFrame(filas).sort_values("peso", key=lambda s: s.abs()).tail(n)

        if hasattr(modelo_final, "feature_importances_"):
            importances = modelo_final.feature_importances_
            idx = importances.argsort()[::-1][:n]
            return pd.DataFrame({
                "feature": [all_feats[i] if i < len(all_feats) else f"feat_{i}" for i in idx],
                "peso": [float(importances[i]) for i in idx],
                "detalle": ["Importancia global"] * len(idx),
                "color": ["#2166ac"] * len(idx),
            }).sort_values("peso")

        return pd.DataFrame()

    def _tabla_metricas_html(self) -> str:
        filas = []
        for clase in self._le.classes_:
            met = self._reporte_oof.get(clase, {})
            filas.append({
                "clase": clase,
                "precision": round(met.get("precision", 0), 3),
                "recall": round(met.get("recall", 0), 3),
                "f1": round(met.get("f1-score", 0), 3),
                "soporte": int(met.get("support", 0)),
            })
        return pd.DataFrame(filas).to_html(index=False, classes="tabla compacta", border=0)

    def _tabla_predicciones_html(self) -> str:
        if self._df_train.empty or len(self._y_pred) != len(self._df_train):
            return ""
        df_pred = self._df_train[["id", "titulo", "fuente"]].copy()
        df_pred["prediccion_oof"] = self._le.inverse_transform(self._y_pred)
        df_pred["resultado"] = np.where(
            df_pred["fuente"] == df_pred["prediccion_oof"],
            "correcto",
            "error",
        )
        df_pred["titulo"] = df_pred["titulo"].fillna("").astype(str).str.slice(0, 120)
        df_pred = df_pred.sort_values(["resultado", "fuente", "prediccion_oof", "id"])
        return df_pred.to_html(index=False, classes="tabla", border=0, table_id="predicciones")

    @staticmethod
    def _envolver_panel_html(grafico: str, metricas: str, predicciones: str) -> str:
        return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panel interactivo caso 4</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f6f8fa;
    }}
    main {{
      max-width: 1220px;
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
    h1 {{
      font-size: 24px;
    }}
    h2 {{
      font-size: 17px;
    }}
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
    .compacta {{
      max-width: 640px;
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
      <h1>Panel interactivo del clasificador</h1>
      {grafico}
    </section>
    <section>
      <h2>Métricas out-of-fold por clase</h2>
      {metricas}
    </section>
    <section>
      <h2>Predicciones out-of-fold</h2>
      <input id="filtro" type="search" placeholder="Filtrar por id, título, fuente o resultado">
      <div class="table-wrap">{predicciones}</div>
    </section>
  </main>
  <script>
    const filtro = document.getElementById("filtro");
    const tabla = document.getElementById("predicciones");
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
