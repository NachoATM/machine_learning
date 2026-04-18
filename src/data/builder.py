"""CorpusBuilder: construye el DataFrame estructurado con feature engineering."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Transforma la lista de documentos crudos en un DataFrame analizable."""

    # ── Atributos de clase ─────────────────────────────────────────────────────

    KEYWORDS_FUENTE: dict[str, list[str]] = {
        "Interior": [
            "guardia civil", "policia nacional", "interior", "seguridad del estado",
            "direccion general de seguridad", "delegado del gobierno", "ppe",
        ],
        "Defensa": [
            "consejo supremo de justicia militar", "ejercito", "capitania general",
            "cni", "cesid", "defensa", "juzgado togado", "causa 2/81", "causa 281",
            "brigada", "division", "regimiento", "cuartel", "estado mayor",
            "teniente coronel", "almirante", "infanteria",
        ],
        "Exteriores": [
            "embajada", "legacion", "diplomatico", "exteriores", "asuntos exteriores",
            "ambassador", "telegram", "despacho", "agregado", "consul",
        ],
    }

    PALETA_FUENTE: dict[str, str] = {
        "Defensa": "#2166ac",
        "Interior": "#d73027",
        "Exteriores": "#1a9850",
        "No determinado": "#999999",
    }

    PALETA_PERIODO: dict[str, str] = {
        "Pre-golpe": "#fdae61",
        "23-F (1981)": "#d73027",
        "Proceso judicial": "#4393c3",
        "Post-proceso": "#74add1",
        "Desconocido": "#cccccc",
    }

    # ── Constructor ────────────────────────────────────────────────────────────

    def __init__(self, raw_docs: list[dict]) -> None:
        self.raw_docs = raw_docs

    # ── API pública ────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """Construye y devuelve el DataFrame con todas las features derivadas."""
        records: list[dict] = []
        for doc in self.raw_docs:
            records.append(self._build_record(doc))
        df = pd.DataFrame(records)
        logger.info("DataFrame construido: %d docs - %d variables", *df.shape)
        return df

    # ── Métodos privados: feature engineering ──────────────────────────────────

    def _build_record(self, doc: dict) -> dict:
        titulo = doc.get("titulo") or doc.get("titulo_listing") or ""
        resumen = doc.get("resumen") or ""
        texto_ocr = doc.get("texto_ocr") or ""
        personas: list[str] = doc.get("personas") or []
        lugares: list[str] = doc.get("lugares") or []
        keywords: list[str] = doc.get("keywords") or []

        paginas = self._parse_paginas(doc.get("paginas") or doc.get("paginas_listing"))
        anio = self._extraer_anio(titulo)
        fuente = self._inferir_fuente(titulo, resumen, keywords)

        n_chars = len(texto_ocr)
        n_pals = len(texto_ocr.split()) if texto_ocr else 0
        n_per = len(personas)
        n_lug = len(lugares)
        ttr = self._riqueza_lexica(texto_ocr)
        densidad = (n_per + n_lug) / paginas if paginas and paginas > 0 else np.nan
        compres = len(resumen) / n_chars if n_chars > 0 else np.nan

        return {
            "id": doc["id"],
            "titulo": titulo,
            "fuente": fuente,
            "anio": anio,
            "periodo": self._clasificar_periodo(anio),
            "paginas": paginas,
            "resumen": resumen,
            "personas": personas,
            "lugares": lugares,
            "keywords": keywords,
            "texto_ocr": texto_ocr,
            "n_chars_ocr": n_chars,
            "n_palabras_ocr": n_pals,
            "n_personas": n_per,
            "n_lugares": n_lug,
            "n_keywords": len(keywords),
            "riqueza_lexica": ttr,
            "densidad_entidades": round(densidad, 4) if not np.isnan(densidad) else np.nan,
            "compresion_resumen": round(compres, 4) if not np.isnan(compres) else np.nan,
        }

    def _inferir_fuente(self, titulo: str, resumen: str, keywords: Sequence[str]) -> str:
        texto = " ".join(filter(None, [titulo, resumen, " ".join(keywords)])).lower()
        texto = self._quitar_tildes(texto)
        for fuente, claves in self.KEYWORDS_FUENTE.items():
            if any(k in texto for k in claves):
                return fuente
        return "No determinado"

    def _extraer_anio(self, titulo: str) -> int | None:
        m = re.search(r"\b(19[7-9]\d|20[012]\d)\b", titulo or "")
        return int(m.group(1)) if m else None

    def _clasificar_periodo(self, anio: int | None) -> str:
        if anio is None:
            return "Desconocido"
        if anio < 1981:
            return "Pre-golpe"
        if anio == 1981:
            return "23-F (1981)"
        if anio <= 1983:
            return "Proceso judicial"
        return "Post-proceso"

    def _riqueza_lexica(self, texto: str) -> float:
        tokens = re.findall(r"\b[a-z]{3,}\b", (texto or "").lower())
        return round(len(set(tokens)) / len(tokens), 4) if tokens else np.nan

    # ── Utilidades ─────────────────────────────────────────────────────────────

    @staticmethod
    def _quitar_tildes(texto: str) -> str:
        for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ü", "u"), ("ñ", "n")]:
            texto = texto.replace(a, b)
        return texto

    @staticmethod
    def _parse_paginas(raw) -> float:
        try:
            s = str(raw).strip()
            return int(s) if s.isdigit() else np.nan
        except Exception:
            return np.nan
