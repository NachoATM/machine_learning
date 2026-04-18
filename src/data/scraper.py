"""Scraper23F: extrae metadatos y texto OCR del buscador oficial de RTVE."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Scraper23F:
    """Scraper del buscador 23fbuscador.rtve.es con caché local en JSON."""

    def __init__(
        self,
        base_url: str = "https://23fbuscador.rtve.es",
        raw_path: str | Path = "data/documentos_23f_raw.json",
        delay: float = 0.4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.raw_path = Path(raw_path)
        self.delay = delay
        self._session: requests.Session | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    def scrape(self, force_refresh: bool = False) -> list[dict]:
        """Devuelve la lista de documentos scrapeados. Usa caché si existe."""
        if self.raw_path.exists() and not force_refresh:
            logger.info("Cargando corpus desde caché: %s", self.raw_path)
            with open(self.raw_path, encoding="utf-8") as f:
                return json.load(f)

        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._session = self._build_session()

        logger.info("Obteniendo listado de documentos...")
        doc_list = self._scrape_listado()
        logger.info("%d documentos en el listado", len(doc_list))

        resultados: list[dict] = []
        for i, doc in enumerate(doc_list, 1):
            logger.debug("Detalle [%d/%d] id=%s", i, len(doc_list), doc["id"])
            detalle = self._scrape_detalle(doc["id"])
            merged = {**doc, **detalle}
            if not merged.get("titulo"):
                merged["titulo"] = merged.get("titulo_listing", "")
            resultados.append(merged)
            time.sleep(self.delay)

        logger.info("Scraping completado: %d documentos", len(resultados))
        with open(self.raw_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        logger.info("Datos guardados en %s", self.raw_path)
        return resultados

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (proyecto-academico ML-MBDS)"
        return s

    def _get_soup(self, url: str, retries: int = 3) -> BeautifulSoup | None:
        for attempt in range(retries):
            try:
                r = self._session.get(url, timeout=30)
                r.raise_for_status()
                return BeautifulSoup(r.text, "lxml")
            except Exception as exc:
                if attempt == retries - 1:
                    logger.error("Error al obtener %s: %s", url, exc)
                    return None
                time.sleep(1.0)
        return None

    def _scrape_listado(self) -> list[dict]:
        url = f"{self.base_url}/?page_size=200&page=1"
        soup = self._get_soup(url)
        if soup is None:
            return []

        docs: list[dict] = []
        for row in soup.select("table tbody tr"):
            link = row.select_one("a.link-button")
            if not link:
                continue
            match = re.search(r"/document/ocr/(\d+)", link["href"])
            if not match:
                continue
            cells = row.find_all("td")
            docs.append(
                {
                    "id": int(match.group(1)),
                    "titulo_listing": link.get_text(strip=True),
                    "paginas_listing": cells[1].get_text(strip=True) if len(cells) > 1 else None,
                    "resumen_corto": cells[3].get_text(strip=True) if len(cells) > 3 else None,
                }
            )
        return docs

    def _scrape_detalle(self, doc_id: int) -> dict:
        url = f"{self.base_url}/document/ocr/{doc_id}"
        soup = self._get_soup(url)
        if soup is None:
            return {}

        # Título
        h2 = soup.select_one("header.page-header h2")
        titulo = h2.get_text(strip=True) if h2 else None

        # Páginas desde el grid de metadatos
        paginas: int | None = None
        grid = soup.select_one("div.detail-grid")
        if grid:
            for div in grid.find_all("div"):
                m = re.search(r"P.ginas[:\s]+(\d+)", div.get_text())
                if m:
                    paginas = int(m.group(1))
                    break

        # Resumen completo
        resumen_el = soup.select_one("section.detail-section p.text-box")
        resumen = resumen_el.get_text(strip=True) if resumen_el else None

        # Entidades por categoría
        personas: list[str] = []
        lugares: list[str] = []
        keywords: list[str] = []
        for card in soup.select("article.tag-group-card"):
            chips = [c.get_text(strip=True) for c in card.select("span.tag-chip")]
            clases = card.get("class", [])
            if "tag-group-people" in clases:
                personas = chips
            elif "tag-group-places" in clases:
                lugares = chips
            elif "tag-group-keywords" in clases:
                keywords = chips

        # Texto OCR completo
        pre = soup.select_one("pre.text-box-large")
        texto_ocr = pre.get_text(strip=True) if pre else None

        return {
            "titulo": titulo,
            "paginas": paginas,
            "resumen": resumen,
            "personas": personas,
            "lugares": lugares,
            "keywords": keywords,
            "texto_ocr": texto_ocr,
        }
