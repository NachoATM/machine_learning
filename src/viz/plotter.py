"""Plotter: helper compartido de visualización para todos los casos."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


class Plotter:
    """Centraliza estilos, paletas y utilidades de texto para todos los casos."""

    _STOPWORDS_ES: frozenset[str] = frozenset(
        {
            "de", "la", "el", "en", "y", "a", "los", "del", "se", "las", "por",
            "con", "una", "para", "un", "al", "es", "su", "que", "lo", "le",
            "como", "mas", "pero", "sus", "o", "sin", "sobre", "este", "ya",
            "entre", "cuando", "todo", "esta", "ser", "son", "dos", "tambien",
            "fue", "era", "han", "ha", "muy", "si", "hay", "puede", "asi",
            "desde", "donde", "uno", "toda", "sido", "hasta", "ante", "mismo",
            "dicho", "dicha", "dichos", "dichas", "durante", "dentro", "cual",
            "parte", "tanto", "vez", "bien", "aunque", "ano", "anos", "fecha",
            "numero", "folio", "paginas", "pagina", "que", "otro", "otros",
            "otras", "diez", "tres", "cuatro", "cinco", "seis", "siete",
        }
    )

    def __init__(self, fig_dir: str | Path = "figuras") -> None:
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def paleta_fuente(self) -> dict[str, str]:
        return {
            "Defensa": "#2166ac",
            "Interior": "#d73027",
            "Exteriores": "#1a9850",
            "No determinado": "#999999",
        }

    @property
    def paleta_periodo(self) -> dict[str, str]:
        return {
            "Pre-golpe": "#fdae61",
            "23-F (1981)": "#d73027",
            "Proceso judicial": "#4393c3",
            "Post-proceso": "#74add1",
            "Desconocido": "#cccccc",
        }

    # ── API pública ────────────────────────────────────────────────────────────

    def savefig(self, nombre: str, fig) -> Path:
        """Guarda la figura con dpi=150, la cierra y devuelve la ruta."""
        path = self.fig_dir / nombre
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def limpiar_texto(self, texto: str) -> str:
        """Preprocesamiento para TF-IDF: minúsculas, sin tildes, sin stopwords."""
        texto = (texto or "").lower()
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            texto = texto.replace(a, b)
        texto = re.sub(r"[^a-z\s]", " ", texto)
        tokens = [t for t in texto.split() if len(t) > 3 and t not in self._STOPWORDS_ES]
        return " ".join(tokens)

    def nuevo_fig(self, *args, **kwargs):
        """Alias conveniente para plt.subplots con estilo ya aplicado."""
        return plt.subplots(*args, **kwargs)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_style() -> None:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 5),
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "font.family": "DejaVu Sans",
            }
        )
