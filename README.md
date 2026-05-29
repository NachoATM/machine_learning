#  Documentos Desclasificados del 23-F · Machine Learning

> Análisis integral de los **167 documentos desclasificados del 23-F** (RTVE) mediante un pipeline completo de *scraping → corpus → 6 casos de ML, NLP, grafos y visualización*.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/BERTopic-NLP-5B2C6F" />
  <img src="https://img.shields.io/badge/NetworkX-grafos-2C5BB4" />
  <img src="https://img.shields.io/badge/Plotly-interactivo-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Universidad_de_Navarra-2025--2026-002E5D" />
</p>

**Equipo:** Ander Corta Arrieta · Jose Ignacio Esteban González · Alejandro Fernández Rubio

---

##  Objetivo

Aplicar de forma integrada técnicas de **Machine Learning, NLP, análisis de redes y visualización** sobre un dataset documental real: los archivos del intento de golpe de Estado del 23 de febrero de 1981, desclasificados y publicados por RTVE.

 **Fuente:** [Buscador 23-F · RTVE](https://www.rtve.es/noticias/23f-desclasificados/buscador-23f/) · [La Moncloa](https://www.lamoncloa.gob.es/consejodeministros/paginas/desclasificacion-documentos-23f.aspx)

---

##  Flujo del proyecto

```
    23fbuscador.rtve.es
          │  Scraper23F  (requests + BeautifulSoup, caché JSON)
          ▼
    documentos_23f_raw.json   (167 documentos · OCR + metadatos)
          │  CorpusBuilder  (feature engineering: fuente, período, riqueza…)
          ▼
    DataFrame estructurado
          │
          ├─▶ Caso 1   EDA + Clustering
          ├─▶ Caso 2   Grafo de actores
          ├─▶ Caso 3   Topic Modeling
          ├─▶ Caso 4   Clasificador supervisado
          ├─▶ Caso 5   Espacio-temporal + anomalías
          └─▶ Caso 6   Explorador interactivo
                         │
                         ▼
               figuras/   ·    outputs/
```

> Todo se orquesta desde **`main.ipynb`**; la lógica vive en `src/` y cada caso hereda de una interfaz común `BaseCaso` (`run()` + `export()`).

---

##  Los 6 casos de análisis

| # | Caso | Qué hace | Técnicas |
|---|------|----------|----------|
| **1** |  **EDA + Clustering** | Radiografía del corpus (fuente, páginas, período, riqueza léxica) y agrupación documental | TF-IDF · KMeans · PCA · Elbow/Silhouette |
| **2** |  **Grafo de actores** | Red de co-menciones de personas; comunidades y nodos clave por período | NetworkX · Louvain · betweenness |
| **3** |  **Topic Modeling** | Temas latentes del corpus y su evolución temporal | BERTopic *(fallback NMF)* · Sentence-Transformers · UMAP |
| **4** |  **Clasificador** | Predice el ministerio de origen (Defensa / Interior / Exteriores) | LogisticRegression · RandomForest · LinearSVC + CV |
| **5** |  **Espacio-temporal** | Serie temporal, mapa geográfico de menciones y detección de anomalías | IsolationForest · PCA · geocoding |
| **6** |  **Explorador interactivo** | "Sala de investigación" HTML para navegar documentos, actores y tópicos | Grafos + TF-IDF + HTML/JS autónomo |

---

##  Estructura

```
machine_learning-T-Nacho/
├── main.ipynb            # Notebook orquestador
├── requirements.txt
├── data/
│   └── documentos_23f_raw.json   # corpus crudo (167 docs)
├── src/
│   ├── base.py           # interfaz BaseCaso (run / export / summary)
│   ├── data/
│   │   ├── scraper.py    # Scraper23F  → JSON
│   │   └── builder.py    # CorpusBuilder → DataFrame
│   ├── casos/
│   │   ├── caso1_eda.py
│   │   ├── caso2_grafos.py
│   │   ├── caso3_topics.py
│   │   ├── caso4_clasificador.py
│   │   ├── caso5_spatiotemporal.py
│   │   └── caso6_explorador.py
│   └── viz/plotter.py    # estilos y figuras comunes
├── figuras/              # PNG + paneles HTML interactivos
└── outputs/              # CSV / JSON / modelos exportados
```

---

##  Uso

```bash
pip install -r requirements.txt

# Abrir y ejecutar el orquestador de principio a fin
jupyter notebook main.ipynb
```

```python
# Patrón común de cada caso
from src.casos.caso1_eda import EDACorpus

caso1 = EDACorpus(df, output_dir="outputs", fig_dir="figuras")
results1 = caso1.run()     # ejecuta el análisis → métricas
caso1.export()             # persiste figuras y datos
```

>  El scraper usa **caché local**: si `documentos_23f_raw.json` ya existe, no vuelve a descargar. Los casos 3 y 6 generan **paneles HTML interactivos** dentro de `figuras/`.

---

##  Stack

`Python` · `pandas` · `scikit-learn` · `BERTopic` · `sentence-transformers` · `NetworkX` · `matplotlib` · `seaborn` · `plotly` · `wordcloud` · `BeautifulSoup`
