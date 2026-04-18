import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseCaso(ABC):
    """Interfaz común para los cinco casos de análisis del proyecto 23-F."""

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str | Path = "outputs",
        fig_dir: str | Path = "figuras",
    ) -> None:
        self.df = df
        self.output_dir = Path(output_dir)
        self.fig_dir = Path(fig_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── Métodos abstractos ───

    @abstractmethod
    def run(self) -> dict:
        """Ejecuta el análisis completo y devuelve un diccionario de métricas."""

    @abstractmethod
    def export(self) -> None:
        """Persiste los resultados del caso en disco."""

    # ── Métodos concretos ───

    def summary(self) -> str:
        """Devuelve un resumen en texto de los resultados principales."""
        if not self._results:
            return f"[{self.__class__.__name__}] run() no ejecutado todavía."
        lines = [f"=== {self.__class__.__name__} ==="]
        for k, v in self._results.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _savefig(self, nombre: str, fig) -> None:
        """Guarda una figura con dpi=150 y la cierra para liberar memoria."""
        path = self.fig_dir / nombre
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.logger.info("Figura guardada: %s", path)

    def _save_json(self, nombre: str, data: dict) -> None:
        """Devuelve un json"""
        path = self.output_dir / nombre
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info("JSON guardado: %s", path)

    def _save_csv(self, nombre: str, df: pd.DataFrame) -> None:
        """Devuelve un csv"""
        path = self.output_dir / nombre
        df.to_csv(path, index=False, encoding="utf-8")
        self.logger.info("CSV guardado: %s", path)
