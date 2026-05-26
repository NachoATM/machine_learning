from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.base import BaseCaso

logger = logging.getLogger(__name__)


class ExpedienteInteractivo(BaseCaso):


    UMBRAL_NODOS = 2    
    N_TOPICOS    = 6    
    LAYOUT_SEED  = 42

    _GENERICOS = {
        "no consta", "no determinado", "sr", "sra", "presidente", "tribunal",
        "el tribunal", "defensores", "procesados", "fiscal", "el fiscal",
        "ministerio fiscal", "testigo", "letrado", "letrados", "acusado",
        "sala", "presidente de la sala", "el presidente de la sala",
        "presidente del tribunal", "tribunal supremo", "relator",
        "secretario relator", "prensa", "consejo", "mandos", "diputados",
        "gobierno", "la corona", "capitan general", "secretario general",
        "presidente del consejo", "militares procesados",
        "tenientes de la guardia civil", "el director general",
        "miembros del servicio", "sectores y bases", "las fuerzas armadas",
        "el pueblo", "el pueblo espanol", "pueblo espanol", "espana",
        "policia nacional", "jefatura superior de policia de madrid",
        "ministerio del interior", "parlamento espanol",
        "congreso de los estados unidos", "consejero togado juez especial",
        "el teniente general presidente", "comandante", "coronel",
        "el ministro de asuntos exteriores",
        "senor ministro de asuntos exteriores",
        "capitan general de valencia",
    }


    _ALIAS: dict[str, str] = {}

    @classmethod
    def _init_alias(cls) -> None:
        """Registra todas las variantes en _ALIAS (se llama una sola vez)."""
        if cls._ALIAS:
            return
        def reg(canon: str, variantes: list[str]) -> None:
            for v in variantes:
                cls._ALIAS[v] = canon

        reg("antonio tejero", [
            "tejero", "teniente coronel tejero", "el teniente coronel tejero",
            "teniente coronel de la guardia civil tejero molina",
            "teniente coronel don antonio tejero molina",
        ])
        reg("alfonso armada", [
            "general armada", "el general armada", "armada", "alfonso armada comyn",
        ])
        reg("milans del bosch", [
            "teniente general milans del bosch", "general milans del bosch",
            "general milans", "teniente general milans",
            "jaime milans del bosch y ussia",
        ])
        reg("juan carlos i", [
            "s.m. el rey", "el rey", "rey", "king juan carlos", "s.m.",
            "su majestad el rey", "rey juan carlos",
        ])
        reg("jose luis cortina", [
            "comandante cortina", "cte. cortina", "jose cortina prieto",
        ])
        reg("gomez iglesias", [
            "capitan gomez iglesias",
            "capitan de la guardia civil gomez iglesias",
        ])
        reg("coronel ibanez",   ["coronel ibanez", "coronel ibanez ingles"])
        reg("garcia carres",    ["garcia carres", "carres"])
        reg("ramon hermosilla", ["hermosilla", "ramon hermosilla martin"])
        reg("perez-llorca",     ["jose pedro perez-llorca", "jose pedro perez llorca"])
        reg("camilo menendez",  [
            "capitan de navio menendez",
            "capitan de navio d. camilo menendez vives",
        ])
        reg("ministro de defensa", [
            "sr. ministro de defensa", "senor ministro de defensa",
            "ministro de defensa", "minisdef",
        ])
        reg("ronald reagan", ["presidente reagan", "ronald reagan"])

    _COORDS: dict[str, tuple[float, float]] = {
        "madrid": (40.4168, -3.7038), "barcelona": (41.3851, 2.1734),
        "valencia": (39.4699, -0.3763), "zaragoza": (41.6488, -0.8891),
        "sevilla": (37.3891, -5.9845), "bilbao": (43.2630, -2.9350),
        "la coruna": (43.3623, -8.4115), "toledo": (39.8628, -4.0273),
        "alicante": (38.3452, -0.4810), "vitoria": (42.8467, -2.6727),
        "san sebastian": (43.3183, -1.9812), "malaga": (36.7213, -4.4213),
        "cadiz": (36.5271, -6.2886),
        "congreso de los diputados": (40.4165, -3.6972),
        "congreso": (40.4165, -3.6972),
        "palacio de la zarzuela": (40.4532, -3.7874),
        "palacio de la moncloa": (40.4386, -3.7152),
        "moncloa": (40.4386, -3.7152),
        "palace": (40.4168, -3.7038),
        "washington": (38.9072, -77.0369), "paris": (48.8566, 2.3522),
        "london": (51.5074, -0.1278), "bonn": (50.7374, 7.0982),
        "lisboa": (38.7169, -9.1395), "roma": (41.9028, 12.4964),
    }
    _NICE: dict[str, str] = {
        "madrid": "Madrid", "barcelona": "Barcelona", "valencia": "Valencia",
        "zaragoza": "Zaragoza", "sevilla": "Sevilla", "bilbao": "Bilbao",
        "la coruna": "La Coruña", "toledo": "Toledo", "alicante": "Alicante",
        "vitoria": "Vitoria", "san sebastian": "San Sebastián",
        "malaga": "Málaga", "cadiz": "Cádiz",
        "congreso de los diputados": "Congreso de los Diputados",
        "palacio de la zarzuela": "Palacio de la Zarzuela",
        "palacio de la moncloa": "Palacio de la Moncloa",
        "washington": "Washington", "paris": "París", "london": "Londres",
        "bonn": "Bonn", "lisboa": "Lisboa", "roma": "Roma",
    }
    _ALIAS_LUGAR: dict[str, str] = {
        "congreso": "congreso de los diputados",
        "moncloa": "palacio de la moncloa",
        "palace": "madrid",
    }



    _HTML_CSS: str = """
:root{--bg:#0d1014;--ink:#11151b;--ink2:#161b23;--ink3:#1d2530;--line:#28313f;
  --tx:#e8eaef;--tx2:#8b95a5;--tx3:#5a6373;--gold:#d4a23c;--red:#c0392b;--paper:#e8e2d0;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);overflow-x:hidden;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.mono{font-family:"SF Mono",ui-monospace,"Cascadia Code",Menlo,monospace;}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);}
#app{display:grid;grid-template-columns:220px 1fr 304px;min-height:100vh;}
/* cabecera */
header{grid-column:1/4;background:var(--ink);border-bottom:1px solid var(--line);
  padding:11px 20px;display:flex;align-items:center;gap:14px;}
.stamp{font-family:"SF Mono",monospace;font-size:10px;letter-spacing:1.5px;color:var(--red);
  border:1.5px solid var(--red);padding:3px 8px;border-radius:3px;transform:rotate(-3deg);
  font-weight:700;}
header h1{font-size:14px;font-weight:600;letter-spacing:2px;}
header h1 .y{color:var(--gold);}
.hchips{display:flex;gap:7px;margin-left:6px;}
.hchip{font-family:"SF Mono",monospace;font-size:10px;color:var(--tx2);
  background:var(--ink3);border:1px solid var(--line);padding:3px 8px;border-radius:11px;}
.hright{margin-left:auto;font-family:"SF Mono",monospace;font-size:10px;
  letter-spacing:1px;color:var(--tx3);}
/* sidebar actores */
#actors{background:var(--ink);border-right:1px solid var(--line);
  height:calc(100vh - 47px);overflow-y:auto;padding:12px 9px;}
.sec-h{font-family:"SF Mono",monospace;font-size:9px;letter-spacing:1.2px;color:var(--tx3);
  margin:4px 4px 9px;}
.afind{width:100%;background:var(--ink3);border:1px solid var(--line);color:var(--tx);
  padding:7px 9px;border-radius:5px;font-size:11px;margin-bottom:11px;
  font-family:"SF Mono",monospace;}
.arow{display:flex;align-items:center;gap:7px;padding:6px 7px;border-radius:5px;
  cursor:pointer;margin-bottom:2px;border:1px solid transparent;}
.arow:hover{background:var(--ink3);}
.arow.on{background:var(--ink3);border-color:var(--gold);}
.adot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.aname{font-size:11px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.apr{font-family:"SF Mono",monospace;font-size:8px;color:var(--tx3);}
.aper{font-size:7px;padding:1px 4px;border-radius:6px;white-space:nowrap;}
/* centro */
#main{height:calc(100vh - 47px);overflow-y:auto;padding:18px 22px;}
.empty-c{text-align:center;color:var(--tx3);margin-top:80px;}
.empty-c .big{font-size:34px;margin-bottom:14px;opacity:.35;}
.empty-c .mono{font-size:12px;line-height:1.9;}
/* ficha cabecera */
.fhead{display:flex;align-items:center;gap:14px;margin-bottom:18px;}
.favatar{width:54px;height:54px;border-radius:8px;background:var(--ink3);border:1px solid var(--line);
  display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;
  color:var(--gold);flex-shrink:0;}
.fname{font-size:21px;font-weight:600;}
.fsub{font-family:"SF Mono",monospace;font-size:10px;letter-spacing:1px;color:var(--tx2);
  margin-top:3px;}
.ftags{display:flex;gap:6px;margin-top:7px;}
.ftag{font-size:9px;padding:2px 8px;border-radius:9px;background:var(--ink3);
  border:1px solid var(--line);color:var(--tx2);}
/* metricas */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:18px;}
.mcard{background:var(--ink2);border:1px solid var(--line);border-radius:7px;padding:11px 12px;}
.mcard .ml{font-family:"SF Mono",monospace;font-size:8px;letter-spacing:1px;color:var(--tx3);
  text-transform:uppercase;}
.mcard .mv{font-size:23px;font-weight:600;color:var(--gold);margin-top:5px;}
.mcard .mx{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);margin-top:3px;}
/* secciones */
.block{margin-bottom:20px;}
.blabel{font-family:"SF Mono",monospace;font-size:9px;letter-spacing:1.4px;color:var(--tx3);
  margin-bottom:11px;padding-bottom:5px;border-bottom:1px solid var(--line);}
/* barras centralidad */
.cbar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.cbar-lab{font-family:"SF Mono",monospace;font-size:10px;color:var(--tx2);width:88px;}
.cbar-track{flex:1;height:9px;background:var(--ink3);border-radius:5px;overflow:hidden;}
.cbar-fill{height:100%;border-radius:5px;}
.cbar-val{font-family:"SF Mono",monospace;font-size:10px;color:var(--tx);width:54px;text-align:right;}
/* grafo local */
#localgraph{background:var(--ink2);border:1px solid var(--line);border-radius:8px;}
.ghint{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);text-align:right;
  margin-top:5px;}
/* documentos */
.doc{background:var(--ink2);border:1px solid var(--line);border-radius:6px;padding:9px 11px;
  margin-bottom:6px;cursor:pointer;}
.doc:hover{border-color:var(--gold);}
.doc .dt{font-size:12px;line-height:1.4;}
.doc .dm{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);margin-top:5px;}
.doc .dtag{font-size:9px;padding:1px 7px;border-radius:8px;background:var(--ink3);color:var(--tx2);
  float:right;}
/* panel derecho */
#panel{background:var(--ink);border-left:1px solid var(--line);
  height:calc(100vh - 47px);overflow-y:auto;padding:16px 14px;}
.pempty{color:var(--tx3);font-family:"SF Mono",monospace;font-size:10px;line-height:1.8;
  text-align:center;margin-top:50px;}
/* topicos */
.topic{margin-bottom:13px;}
.topic-h{display:flex;align-items:baseline;gap:7px;}
.topic-id{font-family:"SF Mono",monospace;font-size:10px;color:var(--gold);font-weight:600;}
.topic-name{font-size:11px;font-weight:500;flex:1;}
.topic-pct{font-family:"SF Mono",monospace;font-size:10px;color:var(--tx2);}
.topic-words{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);margin:4px 0 5px;
  line-height:1.5;}
.topic-track{height:5px;background:var(--ink3);border-radius:3px;overflow:hidden;}
.topic-fill{height:100%;background:var(--gold);}
/* timeline */
.tl{position:relative;padding-left:18px;}
.tl::before{content:"";position:absolute;left:4px;top:4px;bottom:4px;width:2px;
  background:var(--line);}
.tl-item{position:relative;margin-bottom:13px;}
.tl-dot{position:absolute;left:-18px;top:2px;width:10px;height:10px;border-radius:50%;
  border:2px solid var(--bg);}
.tl-per{font-family:"SF Mono",monospace;font-size:9px;letter-spacing:.5px;font-weight:600;}
.tl-bar-track{height:6px;background:var(--ink3);border-radius:3px;margin-top:4px;overflow:hidden;}
.tl-bar-fill{height:100%;border-radius:3px;}
.tl-cnt{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);margin-top:3px;}
/* lugares */
.places{display:flex;flex-wrap:wrap;gap:5px;}
.place{font-family:"SF Mono",monospace;font-size:9px;background:var(--ink3);
  border:1px solid var(--line);padding:3px 7px;border-radius:9px;color:var(--tx2);}
.place .pc{color:var(--gold);}
.minimap{background:var(--ink2);border:1px solid var(--line);border-radius:6px;
  padding:5px;margin-top:7px;}
.minimap svg{display:block;width:100%;height:auto;border-radius:4px;}
.ext-list{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:8px;color:var(--tx3);margin-top:6px;line-height:1.5;}
/* conexiones */
.conn-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;cursor:pointer;}
.conn-row:hover .conn-name{color:var(--gold);}
.conn-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.conn-name{font-size:11px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.conn-track{width:80px;height:6px;background:var(--ink3);border-radius:3px;overflow:hidden;}
.conn-fill{height:100%;background:var(--gold);border-radius:3px;}
.conn-w{font-family:"SF Mono",monospace;font-size:9px;color:var(--tx3);width:18px;text-align:right;}
#gtip{position:absolute;pointer-events:none;background:var(--ink);border:1px solid var(--gold);
  border-radius:5px;padding:5px 8px;font-size:10px;display:none;z-index:20;
  font-family:"SF Mono",monospace;}
"""

    _HTML_BODY: str = """<h2 class="sr-only">Sala de investigacion del corpus del 23-F: selecciona un actor de la barra lateral para ver su ficha completa con metricas de centralidad, grafo local de co-menciones, perfil tematico, linea temporal y mapa de lugares.</h2>
<div id="app">
  <header>
    <span class="stamp">CLASIFICADO</span>
    <h1>EXPEDIENTE <span class="y">23-F</span> &middot; SALA DE INVESTIGACION</h1>
    <div class="hchips" id="hStatsWrap"><span class="hchip" id="hStats"></span></div>
    <span class="hright">CASO 6 &middot; ANALISIS INTEGRADO</span>
  </header>

  <div id="actors">
    <div class="sec-h">ACTORES &mdash; ORDENADOS POR PAGERANK</div>
    <input type="text" class="afind" id="actorFind" placeholder="buscar actor...">
    <div id="actorList"></div>
  </div>

  <div id="main">
    <div class="empty-c" id="emptyMain">
      <div class="big">&#9673;</div>
      <div class="mono">SELECCIONA UN ACTOR EN LA BARRA LATERAL<br>
      PARA ABRIR SU EXPEDIENTE COMPLETO</div>
    </div>
    <div id="mainContent" style="display:none;"></div>
  </div>

  <div id="panel">
    <div class="pempty" id="emptyPanel">
      PERFIL TEMATICO<br>CO-MENCIONES<br>LINEA TEMPORAL<br>LUGARES VINCULADOS<br><br>
      &mdash; ESPERANDO SELECCION &mdash;
    </div>
    <div id="panelContent" style="display:none;"></div>
  </div>
</div>
<div id="gtip"></div>"""

    _HTML_JS: str = """
const D=JSON.parse(document.getElementById('data').textContent);
const PERIODS=['Pre-golpe','23-F (1981)','Proceso judicial','Post-proceso','Desconocido'];
const COM=['#d4a23c','#3f7fc4','#3aa882','#c0392b','#9b7fd0','#6a7585'];
const PER={'Pre-golpe':'#e0922a','23-F (1981)':'#d8463f','Proceso judicial':'#3f8fd8',
  'Post-proceso':'#4fc098','Desconocido':'#6a7585'};
const byId={},docById={};
D.nodes.forEach(n=>byId[n.id]=n);
D.docs.forEach(d=>docById[d.id]=d);
let sel=null;

function perColor(p){return PER[p]||'#6a7585';}
function nodeColor(n){return COM[n.comunidad%6];}
function initials(label){
  const w=label.split(' ').filter(x=>x.length>2);
  return ((w[0]||label)[0]+((w[1]||w[0]||'')[0]||'')).toUpperCase();
}
function maxBy(arr,f){return Math.max.apply(null,arr.map(f));}

// ---- sidebar de actores (ranking por pagerank) ----
function buildSidebar(filter){
  const box=document.getElementById('actorList');
  box.innerHTML='';
  let list=D.nodes.slice().sort((a,b)=>b.pagerank-a.pagerank);
  if(filter)list=list.filter(n=>n.label.toLowerCase().includes(filter));
  list.forEach(n=>{
    const row=document.createElement('div');
    row.className='arow'+(sel===n.id?' on':'');
    row.innerHTML='<span class="adot" style="background:'+nodeColor(n)+'"></span>'
      +'<span class="aname">'+n.label+'</span>'
      +'<span class="apr">PR:'+n.pagerank.toFixed(3)+'</span>'
      +'<span class="aper" style="background:'+perColor(n.periodo)+'33;color:'
      +perColor(n.periodo)+'">'+n.periodo.replace(' (1981)','').replace('Proceso judicial','Proceso')+'</span>';
    row.onclick=()=>select(n.id);
    box.appendChild(row);
  });
}

// ---- grafo local en canvas ----
let lg,lgx;
function drawLocalGraph(n){
  const cv=document.getElementById('localGraph');
  const w=cv.clientWidth,h=300;
  const dpr=window.devicePixelRatio||1;
  cv.width=w*dpr;cv.height=h*dpr;cv.style.height=h+'px';
  const c=cv.getContext('2d');
  c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,w,h);
  // co-menciones
  const conns=[];
  D.edges.forEach(e=>{
    if(e.source===n.id)conns.push([byId[e.target],e.weight]);
    else if(e.target===n.id)conns.push([byId[e.source],e.weight]);
  });
  conns.sort((a,b)=>b[1]-a[1]);
  const show=conns.slice(0,10);
  const cx=w/2,cy=h/2,R=Math.min(w,h)/2-46;
  // posiciones radiales
  const pts=show.map((cn,i)=>{
    const ang=(i/show.length)*Math.PI*2 - Math.PI/2;
    return {node:cn[0],w:cn[1],x:cx+Math.cos(ang)*R,y:cy+Math.sin(ang)*R};
  });
  lg={center:{x:cx,y:cy,node:n},pts:pts};
  const mw=maxBy(show,x=>x[1])||1;
  // aristas
  pts.forEach(p=>{
    c.strokeStyle='rgba(212,162,60,0.30)';
    c.lineWidth=1+(p.w/mw)*3.5;
    c.beginPath();c.moveTo(cx,cy);c.lineTo(p.x,p.y);c.stroke();
  });
  // nodos vecinos
  pts.forEach(p=>{
    c.beginPath();c.arc(p.x,p.y,15,0,7);
    c.fillStyle=nodeColor(p.node);c.globalAlpha=.9;c.fill();c.globalAlpha=1;
    c.lineWidth=1.5;c.strokeStyle='#11151b';c.stroke();
    c.fillStyle='#11151b';c.font='600 9px monospace';c.textAlign='center';
    c.fillText(initials(p.node.label),p.x,p.y+3);
    c.fillStyle='#8b95a5';c.font='9px monospace';
    const lab=p.node.label.length>16?p.node.label.slice(0,15)+'...':p.node.label;
    c.fillText(lab,p.x,p.y+27);
  });
  // nodo central
  c.beginPath();c.arc(cx,cy,21,0,7);
  c.fillStyle=nodeColor(n);c.fill();
  c.lineWidth=2.5;c.strokeStyle='#d4a23c';c.stroke();
  c.fillStyle='#11151b';c.font='700 12px monospace';c.textAlign='center';
  c.fillText(initials(n.label),cx,cy+4);
}

// ---- mini-mapa ----
function miniMap(lugares){
  if(!lugares||!lugares.length)
    return '<div class="topic-words">Sin lugares geolocalizados.</div>';
  // separar nacionales / extranjeros
  const esp=lugares.filter(l=>l.esp!==false), ext=lugares.filter(l=>l.esp===false);
  // bounding box de la Espana peninsular
  const LAT0=35.8,LAT1=43.9,LON0=-9.6,LON1=3.5,w=270,h=176;
  function proj(lat,lon){
    return {x:(lon-LON0)/(LON1-LON0)*w, y:(LAT1-lat)/(LAT1-LAT0)*h};
  }
  // contorno simplificado de la peninsula iberica (parte espanola), en lat/lon
  const OUTLINE=[[43.79,-7.87],[43.55,-5.70],[43.40,-3.00],[43.46,-1.78],
    [42.90,-1.30],[42.34,0.70],[41.26,1.95],[41.22,2.19],[40.62,0.52],
    [39.88,0.21],[38.84,-0.10],[38.15,-0.65],[37.56,-0.69],[37.23,-1.63],
    [36.74,-2.12],[36.69,-3.52],[36.51,-4.62],[36.18,-5.35],[36.01,-5.61],
    [36.18,-6.16],[37.10,-6.85],[37.20,-7.40],[37.94,-7.41],[39.10,-7.00],
    [39.66,-7.54],[40.09,-6.86],[41.03,-6.86],[41.94,-6.76],[41.88,-8.19],
    [42.58,-8.88],[43.05,-9.28],[43.79,-7.87]];
  let path='';
  OUTLINE.forEach((p,i)=>{
    const q=proj(p[0],p[1]);
    path+=(i?'L':'M')+q.x.toFixed(1)+' '+q.y.toFixed(1)+' ';
  });
  path+='Z';
  // agrupar lugares que caen a menos de 14px (p.ej. todo lo de Madrid)
  const placed=[];
  esp.slice().sort((a,b)=>b.count-a.count).forEach(l=>{
    const q=proj(l.lat,l.lon);
    let g=placed.find(p=>Math.hypot(p.x-q.x,p.y-q.y)<14);
    if(g){ g.items.push(l); g.count+=l.count; }
    else placed.push({x:q.x,y:q.y,count:l.count,items:[l]});
  });
  const mx=placed.length?Math.max.apply(null,placed.map(p=>p.count)):1;
  let dots='';
  placed.forEach((p,i)=>{
    const r=4+Math.sqrt(p.count/mx)*11;
    dots+='<circle cx="'+p.x.toFixed(1)+'" cy="'+p.y.toFixed(1)+'" r="'+r.toFixed(1)
      +'" fill="rgba(212,162,60,0.32)" stroke="#d4a23c" stroke-width="1.2"/>';
    // etiqueta solo del lugar dominante del grupo, alternando arriba/abajo
    const lab=p.items[0].name+(p.items.length>1?' (+'+(p.items.length-1)+')':'');
    const up=(i%2===0);
    dots+='<text x="'+p.x.toFixed(1)+'" y="'+(p.y+(up?-r-3:r+9)).toFixed(1)
      +'" fill="#b8c0cd" font-size="7.5" text-anchor="middle">'+lab+'</text>';
  });
  let html='<div class="minimap"><svg viewBox="0 0 '+w+' '+h+'" role="img" '
    +'aria-label="Mapa de la Espana peninsular con los lugares mencionados en los documentos del actor">'
    +'<rect width="'+w+'" height="'+h+'" fill="#0d1014"/>'
    +'<path d="'+path+'" fill="#1b222d" stroke="#3a4654" stroke-width="1"/>'
    +dots+'</svg></div>';
  if(ext.length){
    html+='<div class="ext-list">Fuera de Espana: '
      +ext.map(l=>l.name+' ('+l.count+')').join(' \u00b7 ')+'</div>';
  }
  return html;
}

// ---- seleccionar actor ----
function select(id){
  sel=id;
  const n=byId[id];
  buildSidebar(document.getElementById('actorFind').value.toLowerCase().trim());
  // rol
  let role='ACTOR DEL CORPUS';
  if(n.betweenness>0.12)role='CONECTOR CLAVE';
  else if(n.pagerank>0.05)role='ACTOR CENTRAL';
  document.getElementById('emptyMain').style.display='none';
  document.getElementById('emptyPanel').style.display='none';
  const M=document.getElementById('mainContent');
  const P=document.getElementById('panelContent');
  M.style.display='block';P.style.display='block';

  // metricas
  const prMax=maxBy(D.nodes,x=>x.pagerank);
  const btMax=maxBy(D.nodes,x=>x.betweenness);
  const dgMax=maxBy(D.nodes,x=>x.degree);
  const conns=[];
  D.edges.forEach(e=>{
    if(e.source===id)conns.push([byId[e.target],e.weight]);
    else if(e.target===id)conns.push([byId[e.source],e.weight]);
  });
  conns.sort((a,b)=>b[1]-a[1]);
  const docs=n.docs.map(i=>docById[i]).filter(Boolean)
    .sort((a,b)=>(a.anio||9999)-(b.anio||9999));

  let h='';
  // cabecera ficha
  h+='<div class="fhead"><div class="favatar">'+initials(n.label)+'</div><div>';
  h+='<div class="fname">'+n.label+'</div>';
  h+='<div class="fsub">'+role+' &middot; COMUNIDAD '+(n.comunidad+1)+'</div>';
  h+='<div class="ftags"><span class="ftag" style="border-color:'+perColor(n.periodo)
    +';color:'+perColor(n.periodo)+'">'+n.periodo+'</span>'
    +'<span class="ftag">'+n.freq+' docs</span>'
    +'<span class="ftag">'+conns.length+' co-menciones</span></div>';
  h+='</div></div>';
  // metricas
  h+='<div class="metrics">';
  h+='<div class="mcard"><div class="ml">PageRank</div><div class="mv">'+n.pagerank.toFixed(3)
    +'</div><div class="mx">'+Math.round(n.pagerank/prMax*100)+'% del maximo</div></div>';
  h+='<div class="mcard"><div class="ml">Betweenness</div><div class="mv">'+n.betweenness.toFixed(3)
    +'</div><div class="mx">'+Math.round(n.betweenness/btMax*100)+'% del maximo</div></div>';
  h+='<div class="mcard"><div class="ml">Degree pond.</div><div class="mv">'+n.degree
    +'</div><div class="mx">'+Math.round(n.degree/dgMax*100)+'% del maximo</div></div>';
  h+='<div class="mcard"><div class="ml">Menciones</div><div class="mv">'+n.n_menciones_lugar
    +'</div><div class="mx">en '+n.freq+' documentos</div></div>';
  h+='</div>';
  // centralidad comparada
  h+='<div class="block"><div class="blabel">CENTRALIDAD COMPARADA</div>';
  [['PageRank',n.pagerank,prMax,'#d4a23c'],
   ['Betweenness',n.betweenness,btMax,'#3f8fd8'],
   ['Degree',n.degree,dgMax,'#3aa882']].forEach(r=>{
    h+='<div class="cbar-row"><span class="cbar-lab">'+r[0]+'</span>'
      +'<span class="cbar-track"><span class="cbar-fill" style="width:'
      +(r[1]/r[2]*100).toFixed(1)+'%;background:'+r[3]+'"></span></span>'
      +'<span class="cbar-val">'+(r[1]<1?r[1].toFixed(3):r[1])+'</span></div>';
  });
  h+='</div>';
  // grafo local
  h+='<div class="block"><div class="blabel">RED DE CO-MENCIONES (GRAFO LOCAL)</div>';
  h+='<canvas id="localGraph" style="width:100%;display:block;"></canvas>';
  h+='<div class="ghint">El nodo central es '+n.label+' &middot; sus 10 co-menciones mas fuertes</div></div>';
  // documentos
  h+='<div class="block"><div class="blabel">DOCUMENTOS DONDE APARECE ('+docs.length+')</div>';
  docs.slice(0,20).forEach(d=>{
    h+='<div class="doc" data-doc="'+d.id+'">'
      +'<span class="dtag">'+d.periodo.replace(' (1981)','')+'</span>'
      +'<div class="dt">'+d.titulo+'</div>'
      +'<div class="dm">'+d.fuente+'  &middot;  '+(d.anio||'s/f')
      +'  &middot;  T'+d.topico+'</div></div>';
  });
  h+='</div>';
  M.innerHTML=h;
  M.scrollTop=0;
  M.querySelectorAll('[data-doc]').forEach(el=>el.onclick=()=>{
    const d=docById[el.dataset.doc];
    sendPrompt('Cuentame mas sobre este documento del corpus 23-F: "'+d.titulo+'"');
  });
  drawLocalGraph(n);

  // ----- panel derecho -----
  let p='';
  // topicos
  p+='<div class="block"><div class="blabel">PERFIL TEMATICO</div>';
  const tTot=n.topic_dist.reduce((s,x)=>s+x.n,0);
  n.topic_dist.slice(0,4).forEach(td=>{
    const pct=Math.round(td.n/tTot*100);
    const words=D.topic_labels[td.t]||'';
    p+='<div class="topic"><div class="topic-h">'
      +'<span class="topic-id">T'+td.t+'</span>'
      +'<span class="topic-name">'+words.split(',')[0]+'</span>'
      +'<span class="topic-pct">'+pct+'%</span></div>';
    p+='<div class="topic-words">'+words+'</div>';
    p+='<div class="topic-track"><div class="topic-fill" style="width:'+pct+'%"></div></div></div>';
  });
  p+='</div>';
  // co-menciones con barras
  p+='<div class="block"><div class="blabel">CO-MENCIONES DIRECTAS</div>';
  const cMax=conns.length?conns[0][1]:1;
  conns.slice(0,7).forEach(c=>{
    p+='<div class="conn-row" data-go="'+c[0].id+'">'
      +'<span class="conn-dot" style="background:'+nodeColor(c[0])+'"></span>'
      +'<span class="conn-name">'+c[0].label+'</span>'
      +'<span class="conn-track"><span class="conn-fill" style="width:'
      +(c[1]/cMax*100)+'%"></span></span>'
      +'<span class="conn-w">'+c[1]+'</span></div>';
  });
  p+='</div>';
  // timeline
  p+='<div class="block"><div class="blabel">LINEA TEMPORAL (DOCS POR PERIODO)</div><div class="tl">';
  const pMax=maxBy(n.periodo_dist,x=>x.n)||1;
  n.periodo_dist.forEach(pd=>{
    const col=perColor(pd.p);
    const active=pd.n>0;
    p+='<div class="tl-item">'
      +'<span class="tl-dot" style="background:'+(active?col:'#28313f')+'"></span>'
      +'<div class="tl-per" style="color:'+(active?col:'#5a6373')+'">'+pd.p+'</div>'
      +'<div class="tl-bar-track"><div class="tl-bar-fill" style="width:'
      +(pd.n/pMax*100)+'%;background:'+col+'"></div></div>'
      +'<div class="tl-cnt">'+pd.n+' documento'+(pd.n===1?'':'s')+'</div></div>';
  });
  p+='</div></div>';
  // lugares
  p+='<div class="block"><div class="blabel">LUGARES VINCULADOS</div>';
  p+='<div class="places">';
  (n.lugares||[]).slice(0,10).forEach(l=>{
    p+='<span class="place">'+l.name+' <span class="pc">'+l.count+'</span></span>';
  });
  p+='</div>';
  p+=miniMap(n.lugares);
  p+='</div>';
  P.innerHTML=p;
  P.scrollTop=0;
  P.querySelectorAll('[data-go]').forEach(el=>el.onclick=()=>select(el.dataset.go));
}

// tooltip grafo local
document.addEventListener('mousemove',e=>{
  const cv=document.getElementById('localGraph');
  if(!cv||!lg)return;
  const r=cv.getBoundingClientRect();
  if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){
    document.getElementById('gtip').style.display='none';return;}
  const mx=e.clientX-r.left,my=e.clientY-r.top;
  let hit=null;
  lg.pts.forEach(p=>{if(Math.hypot(p.x-mx,p.y-my)<16)hit=p;});
  if(Math.hypot(lg.center.x-mx,lg.center.y-my)<22)hit={node:lg.center.node,w:null};
  const tip=document.getElementById('gtip');
  if(hit){
    tip.style.display='block';
    tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+10)+'px';
    tip.textContent=hit.node.label+(hit.w!==null?' ('+hit.w+' docs juntos)':'');
    cv.style.cursor='pointer';
  }else{tip.style.display='none';cv.style.cursor='default';}
});
document.addEventListener('click',e=>{
  const cv=document.getElementById('localGraph');
  if(!cv||!lg)return;
  const r=cv.getBoundingClientRect();
  if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)return;
  const mx=e.clientX-r.left,my=e.clientY-r.top;
  lg.pts.forEach(p=>{if(Math.hypot(p.x-mx,p.y-my)<16)select(p.node.id);});
});

// --- arranque protegido: la lista se construye SI o SI ---
function boot(){
  try{ buildSidebar(''); }catch(err){ console.error('buildSidebar:',err); }
  try{
    const hs=document.getElementById('hStats');
    if(hs)hs.textContent=D.stats.n_actores+' ACTORES  /  '+D.stats.n_docs
      +' DOCUMENTOS  /  '+Object.keys(D.topic_labels).length+' TOPICOS';
  }catch(err){ console.error('hStats:',err); }
  try{
    const af=document.getElementById('actorFind');
    if(af)af.oninput=e=>buildSidebar(e.target.value.toLowerCase().trim());
  }catch(err){ console.error('actorFind:',err); }
  window.addEventListener('resize',()=>{if(sel)drawLocalGraph(byId[sel]);});
}
if(document.readyState==='loading'){
  document.addEventListener('DOMContentLoaded',boot);
}else{ boot(); }
"""

    _HTML_TEMPLATE: str = (
        "<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "<title>Caso 6 — El Expediente Interactivo · 23-F</title>\n"
        "<style>{css}</style>\n</head>\n<body>\n"
        "{body}\n"
        "<script id=\"data\" type=\"application/json\">{data}</script>\n"
        "<script>\n{js}\n</script>\n</body>\n</html>"
    )


    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str | Path = "outputs",
        fig_dir: str | Path = "figuras",
    ) -> None:
        super().__init__(df, output_dir, fig_dir)
        self._init_alias()
        self._graph: nx.Graph | None = None
        self._explorer_data: dict = {}


    def run(self) -> dict:
        logger.info("=== Caso 6: Expediente Interactivo 23-F ===")

        actor_docs, doc_actors = self._normalizar_actores()
        self._graph, pr, btw, node_com, deg = self._construir_grafo(
            actor_docs, doc_actors
        )
        topic_labels = self._calcular_topicos()
        doc_lug = self._geocodificar_lugares()
        self._explorer_data = self._ensamblar_json(
            actor_docs, doc_actors, pr, btw, node_com, deg,
            topic_labels, doc_lug,
        )

        g = self._graph
        self._results = {
            "n_actores": g.number_of_nodes(),
            "n_aristas": g.number_of_edges(),
            "n_comunidades": len(set(node_com.values())),
            "actor_pagerank_top": max(pr, key=pr.get),
            "actor_betweenness_top": max(btw, key=btw.get),
            "densidad": round(nx.density(g), 4),
            "n_docs": len(self.df),
        }
        logger.info("Grafo limpio: %d nodos, %d aristas", g.number_of_nodes(), g.number_of_edges())
        logger.info("Top PageRank: %s", self._results["actor_pagerank_top"])
        logger.info("Top Betweenness: %s", self._results["actor_betweenness_top"])
        return self._results

    def export(self) -> None:
        json_path = self.output_dir / "caso6_datos_explorador.json"
        json_path.write_text(
            json.dumps(self._explorer_data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info("JSON guardado: %s", json_path)

        self._generar_html()

        self._save_json("resumen_caso6.json", self._results)

    def _generar_html(self) -> None:
      
        if not self._explorer_data:
            logger.warning("_generar_html: no hay datos. Ejecuta run() primero.")
            return

        data_json = json.dumps(
            self._explorer_data, ensure_ascii=False, separators=(",", ":")
        )
        html = self._HTML_TEMPLATE.format(
            css=self._HTML_CSS,
            body=self._HTML_BODY,
            data=data_json,
            js=self._HTML_JS,
        )
        html_path = self.fig_dir / "caso6_sala_investigacion.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info("HTML guardado: %s  (%d KB)", html_path, len(html) // 1024)


    def _normalizar_actores(
        self,
    ) -> tuple[defaultdict[str, set], dict]:
        
        actor_docs: defaultdict[str, set] = defaultdict(set)
        doc_actors: dict = {}

        for _, row in self.df.iterrows():
            vistos: set[str] = set()
            for p in (row["personas"] or []):
                n = self._limpia_persona(p)
                if len(n) < 3 or n in self._GENERICOS:
                    continue
                vistos.add(self._canon(n))
            doc_actors[row["id"]] = vistos
            for n in vistos:
                actor_docs[n].add(row["id"])

        return actor_docs, doc_actors

    def _construir_grafo(
        self,
        actor_docs: defaultdict[str, set],
        doc_actors: dict,
    ) -> tuple[nx.Graph, dict, dict, dict, Counter]:
        
        freq = {a: len(d) for a, d in actor_docs.items()}
        nodos = [a for a, c in freq.items() if c >= self.UMBRAL_NODOS]

        co: Counter = Counter()
        for did, acts in doc_actors.items():
            acts_f = [a for a in acts if a in nodos]
            for u, v in combinations(sorted(acts_f), 2):
                co[(u, v)] += 1

        G = nx.Graph()
        for n in nodos:
            G.add_node(n, freq=freq[n])
        for (u, v), w in co.items():
            G.add_edge(u, v, weight=w)
        G.remove_nodes_from(list(nx.isolates(G)))

        pr  = nx.pagerank(G, weight="weight")
        btw = nx.betweenness_centrality(G, weight="weight")
        coms = list(greedy_modularity_communities(G, weight="weight"))
        node_com = {n: i for i, c in enumerate(coms) for n in c}
        deg: Counter = Counter()
        for u, v, w in G.edges(data="weight"):
            deg[u] += w
            deg[v] += w

        return G, pr, btw, node_com, deg

    def _calcular_topicos(self) -> dict[int, str]:
        
        STOP = set(
            "de la el en y a los las un una que se del por con para su al lo como mas pero "
            "sus le ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay "
            "donde quien desde todo nos durante todos uno les ni contra otros ese eso ante "
            "ellos e esto mi antes algunos unos yo otro otras otra tanto esa estos mucho "
            "quienes nada muchos cual sea poco ella the of and to in is it for on as at by "
            "an be or from this with not are was were os em da do no na com uma por dos um "
            "sr sra dn don excmo excma ilmo fr dtor dgs gral tcol tte cap cte".split()
        )
        textos = (
            self.df["resumen"].fillna("") + " " + self.df["texto_ocr"].fillna("")
        ).tolist()
        tfidf = TfidfVectorizer(
            max_features=1500, ngram_range=(1, 2), min_df=3, max_df=0.85,
            sublinear_tf=True, stop_words=list(STOP),
            token_pattern=r"(?u)\b[a-záéíóúñ]{4,}\b",
        )
        X = tfidf.fit_transform(textos)
        km = KMeans(n_clusters=self.N_TOPICOS, random_state=self.LAYOUT_SEED, n_init=10).fit(X)
        self.df = self.df.copy()
        self.df["topico"] = km.labels_
        terms = tfidf.get_feature_names_out()
        return {
            t: ", ".join(terms[i] for i in km.cluster_centers_[t].argsort()[-4:][::-1])
            for t in range(self.N_TOPICOS)
        }

    def _geocodificar_lugares(self) -> dict[int, Counter]:
        
        doc_lug: dict[int, Counter] = {}
        for _, r in self.df.iterrows():
            c: Counter = Counter()
            lugs = r["lugares"] if isinstance(r["lugares"], list) else []
            for lg in lugs:
                k = self._match_lugar(str(lg))
                if k:
                    c[self._ALIAS_LUGAR.get(k, k)] += 1
            doc_lug[int(r["id"])] = c
        return doc_lug

    def _ensamblar_json(
        self,
        actor_docs: defaultdict[str, set],
        doc_actors: dict,
        pr: dict, btw: dict, node_com: dict, deg: Counter,
        topic_labels: dict[int, str],
        doc_lug: dict[int, Counter],
    ) -> dict:
       
        G = self._graph
        pos = nx.spring_layout(G, k=0.9, iterations=200,
                                seed=self.LAYOUT_SEED, weight="weight")

        PERIODOS = ["Pre-golpe", "23-F (1981)", "Proceso judicial",
                    "Post-proceso", "Desconocido"]
        id2periodo = dict(zip(self.df["id"], self.df["periodo"]))
        id2topico  = dict(zip(self.df["id"], self.df["topico"]))

        nodes_out = []
        for n in G.nodes():
            docs = sorted(i for i, a in doc_actors.items() if n in a)
            tp  = Counter(id2topico.get(x) for x in docs)
            prd = Counter(id2periodo.get(x, "Desconocido") for x in docs)
            agg: Counter = Counter()
            for did in docs:
                agg += doc_lug.get(did, Counter())
            lugares = [
                {
                    "name": self._NICE.get(k, k.title()),
                    "lat": self._COORDS[k][0], "lon": self._COORDS[k][1],
                    "count": v,
                    "esp": bool(
                        36 <= self._COORDS[k][0] <= 44
                        and -10 <= self._COORDS[k][1] <= 4
                    ),
                }
                for k, v in agg.most_common()
            ]
            x, y = pos[n]
            nodes_out.append({
                "id": n, "label": self._title(n),
                "x": round(float(x), 4), "y": round(float(y), 4),
                "freq": G.nodes[n]["freq"],
                "pagerank":    round(pr[n], 4),
                "betweenness": round(btw[n], 4),
                "degree":      deg.get(n, 0),
                "comunidad":   node_com[n],
                "periodo":     prd.most_common(1)[0][0],
                "docs":        docs,
                "topic_dist":  [
                    {"t": int(t), "n": c}
                    for t, c in tp.most_common()
                    if t is not None
                ],
                "periodo_dist": [
                    {"p": p, "n": prd.get(p, 0)} for p in PERIODOS
                ],
                "lugares":            lugares,
                "n_menciones_lugar":  sum(l["count"] for l in lugares),
            })

        edges_out = [
            {"source": u, "target": v, "weight": G[u][v]["weight"]}
            for u, v in G.edges()
        ]
        docs_out = [
            {
                "id":         self._si(r["id"]),
                "titulo":     str(r["titulo"])[:140],
                "fuente":     r["fuente"],
                "periodo":    r["periodo"],
                "anio":       self._si(r["anio"]),
                "topico":     self._si(r["topico"]),
                "paginas":    self._si(r["paginas"]),
                "n_personas": self._si(r["n_personas"]),
                "resumen":    (str(r["resumen"]) or "")[:280],
            }
            for _, r in self.df.iterrows()
        ]

        return {
            "nodes":        nodes_out,
            "edges":        edges_out,
            "docs":         docs_out,
            "topic_labels": {str(k): v for k, v in topic_labels.items()},
            "periodos":     PERIODOS,
            "n_comunidades": len(set(node_com.values())),
            "stats": {
                "n_docs":     len(docs_out),
                "n_actores":  len(nodes_out),
                "n_aristas":  len(edges_out),
            },
        }


    @staticmethod
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        s = "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"\s+", " ", s)

    @classmethod
    def _limpia_persona(cls, p: str) -> str:
        n = cls._norm(p.split(":")[0])
        n = re.sub(
            r"^(sr|sra|d|dn|don|excmo|excma|ilmo|tcol|tte|gral)\.?\s+", "", n
        )
        return n.strip()

    @classmethod
    def _canon(cls, n: str) -> str:
        return cls._ALIAS.get(n, n)

    @staticmethod
    def _title(s: str) -> str:
        return " ".join(w.capitalize() for w in s.split())

    @staticmethod
    def _si(v) -> int | None:
        try:
            return int(v) if v == v and v is not None else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def _norm_lugar(cls, v: str) -> str:
        t = unicodedata.normalize("NFKD", str(v).lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()

    @classmethod
    def _match_lugar(cls, lg: str) -> str | None:
        ln = cls._norm_lugar(lg)
        if not ln or ln in {"sala", "radio", "canal"}:
            return None
        for k in sorted(cls._COORDS, key=len, reverse=True):
            kn = cls._norm_lugar(k)
            kt, lt = set(kn.split()), set(ln.split())
            if kn == ln or (kt and kt.issubset(lt)):
                return k
            if len(ln) > 4 and lt and lt.issubset(kt):
                return k
        return None
