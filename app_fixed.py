import os
import math
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None
import tempfile
import io
import zipfile
from datetime import datetime

# Try optional local backends only. No OpenAI keys are required.
LANGCHAIN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBER_AVAILABLE = True
except Exception:
    SBER_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_NN_AVAILABLE = True
except Exception:
    SKLEARN_NN_AVAILABLE = False

if st is not None:
    st.set_page_config(page_title="Plataforma de Simulação Geopolítica (MVP) — RAG", layout="wide")
    st.title("Plataforma de Simulação Geopolítica — MVP (RAG via CSV)")
else:
    # allow importing non-UI components when Streamlit is not installed
    pass

# -----------------------
# Helpers: risk, sim, plot, sentiment
# -----------------------
def compute_risk(tension: float, ndvi: float, naval: float) -> float:
    w_tension = 0.5
    w_naval = 0.3
    w_ndvi = 0.2
    ndvi_inverted = 10 - ndvi
    raw = (w_tension * tension) + (w_naval * naval) + (w_ndvi * ndvi_inverted)
    normalized = (raw / (w_tension*10 + w_naval*10 + w_ndvi*10)) * 100
    return float(np.clip(normalized, 0, 100))

def simulate_evolution(tension: float, ndvi: float, naval: float, steps: int = 12, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # pandas frequency strings may be lowercase depending on pandas version
    times = pd.date_range(start=datetime.now(), periods=steps, freq='h')
    tension_series = tension + np.cumsum(rng.normal(loc=0.0, scale=0.3, size=steps))
    ndvi_series = ndvi + np.cumsum(rng.normal(loc=0.0, scale=0.2, size=steps))
    naval_series = naval + np.cumsum(rng.normal(loc=0.0, scale=0.25, size=steps))
    tension_series = np.clip(tension_series, 0, 10)
    ndvi_series = np.clip(ndvi_series, 0, 10)
    naval_series = np.clip(naval_series, 0, 10)
    df = pd.DataFrame({
        "timestamp": times,
        "Tensão_Diplomática": tension_series,
        "NDVI_Anomalia": ndvi_series,
        "Movimentação_Naval": naval_series,
    })
    df["Risco_Escalada"] = df.apply(lambda r: compute_risk(r["Tensão_Diplomática"], r["NDVI_Anomalia"], r["Movimentação_Naval"]), axis=1)
    df["step"] = np.arange(1, len(df)+1)
    return df

def create_3d_plot(df: pd.DataFrame):
    dfc = df.copy()
    dfc["size"] = (dfc["Risco_Escalada"]/100.0)*15 + 4
    fig = px.scatter_3d(
        dfc,
        x="Tensão_Diplomática",
        y="NDVI_Anomalia",
        z="Movimentação_Naval",
        color="Risco_Escalada",
        size="size",
        size_max=30,
        color_continuous_scale="Turbo",
        hover_data={"timestamp": True, "Risco_Escalada": ':.2f', "step": True},
        labels={
            "Tensão_Diplomática": "Tensão Diplomática",
            "NDVI_Anomalia": "NDVI (Anomalia)",
            "Movimentação_Naval": "Movimentação Naval",
            "Risco_Escalada": "Risco de Escalada (%)"
        },
        title="Proxy 4D (simulado): Evolução do Conflito"
    )
    fig.add_trace(px.line_3d(dfc, x="Tensão_Diplomática", y="NDVI_Anomalia", z="Movimentação_Naval").data[0])
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_colorbar=dict(title="Risco (%)"))
    return fig

def analyze_sentiment_geopolitical(tension: float) -> Tuple[str, str]:
    if tension >= 7:
        return "Hostil", "#d9534f"
    elif tension >= 4:
        return "Neutro", "#f0ad4e"
    else:
        return "Colaborativo", "#5cb85c"

# -----------------------
# RAG: CSV -> docs -> embeddings -> index -> retrieval -> generation
# -----------------------

# 1) Load CSV and build textual docs
def load_csv_as_docs(csv_path: str, id_cols: Optional[List[str]] = None, text_cols: Optional[List[str]] = None, max_chars: int = 1000) -> Tuple[List[str], List[Dict[str,Any]]]:
    df = pd.read_csv(csv_path, low_memory=False)
    id_cols = id_cols or [c for c in df.columns if any(k in c.lower() for k in ["country", "pais", "region", "iso"])]
    if not text_cols:
        possible = [c for c in df.columns if any(k in c.lower() for k in ["desc", "note", "comment", "indicator", "description", "text", "summary"])]
        remainder = [c for c in df.columns if c not in id_cols + possible]
        text_cols = possible[:3] if possible else remainder[:5]
    docs = []
    metadatas = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for idx, row in df.iterrows():
        parts = []
        # Add identifier columns only if value is present
        for c in id_cols:
            if c in df.columns and pd.notna(row[c]):
                parts.append(f"{c}: {row[c]}")
        # Add textual columns only if value is present
        for c in text_cols:
            if c in df.columns and pd.notna(row[c]):
                parts.append(f"{c}: {row[c]}")
        nums = numeric_cols[:6]
        if nums:
            # Safely format numeric summaries
            num_parts = []
            for c in nums:
                try:
                    if pd.notna(row[c]):
                        num_parts.append(f"{c}={float(row[c]):.2f}")
                except Exception:
                    # skip non-numeric entries
                    continue
            num_summary = ", ".join(num_parts)
            if num_summary:
                parts.append(f"Numeric: {num_summary}")
        # Join only existing parts
        doc = " | ".join([str(p) for p in parts]) if parts else ""
        if len(doc) > max_chars:
            doc = doc[:max_chars - 3] + "..."
        docs.append(doc)
        metadatas.append({"index": int(idx)})
    return docs, metadatas

# 2) Embedder class: local sentence-transformers or TF-IDF fallback
class Embedder:
    def __init__(self, openai_api_key: Optional[str] = None, sbert_model_name: str = "all-MiniLM-L6-v2"):
        # External API keys are intentionally unsupported in this local-first version.
        self.openai_api_key = None
        self.sbert_model_name = sbert_model_name
        self._openai_embeddings = None
        self._sbert = None
        if SBER_AVAILABLE:
            try:
                self._sbert = SentenceTransformer(self.sbert_model_name)
            except Exception:
                self._sbert = None
        self._tfidf = None
        if not self._sbert:
            if SKLEARN_NN_AVAILABLE:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self._tfidf = TfidfVectorizer(max_features=2048)
                except Exception:
                    self._tfidf = None
        if not self._sbert and self._tfidf is None:
            raise RuntimeError("Nenhum backend de embeddings disponível. Instale sentence-transformers ou scikit-learn para o fallback TF-IDF local.")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self._openai_embeddings:
            embs = self._openai_embeddings.embed_documents(texts)
            return np.array(embs, dtype=np.float32)
        elif self._sbert:
            embs = self._sbert.encode(texts, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
            return embs.astype(np.float32)
        elif self._tfidf is not None:
            # TF-IDF produces sparse vectors; ensure we fit only once on the corpus and reuse for queries
            if not getattr(self, '_tfidf_fitted', False):
                X = self._tfidf.fit_transform(texts)
                self._tfidf_fitted = True
            else:
                X = self._tfidf.transform(texts)
            arr = X.toarray().astype(np.float32)
            # Optionally reduce dimensionality if too large (skipped here)
            return arr
        else:
            raise RuntimeError("Nenhum embedder disponível.")

# 3) Vector index: FAISS preferível, sklearn fallback
class VectorIndex:
    def __init__(self, embeddings: np.ndarray, use_faiss: bool = True):
        # Keep a safe float32 copy of embeddings
        emb = np.asarray(embeddings, dtype=np.float32).copy()
        self.embeddings = emb
        self.n, self.dim = emb.shape
        # honor caller preference but only enable if FAISS_AVAILABLE
        self.use_faiss = bool(use_faiss) and FAISS_AVAILABLE
        self.index = None
        if self.use_faiss:
            # FAISS expects normalized vectors when using inner-product for cosine-like similarity
            emb_norm = emb.copy()
            faiss.normalize_L2(emb_norm)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(emb_norm)
        else:
            if not SKLEARN_NN_AVAILABLE:
                raise RuntimeError("FAISS não disponível e sklearn.neighbors não está instalado.")
            # Normalize vectors for cosine metric
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_norm = emb / norms
            self.nn = NearestNeighbors(metric="cosine")
            self.nn.fit(emb_norm)
            self._emb_norm = emb_norm

    def query(self, query_emb: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.use_faiss and self.index is not None:
            q_norm = q.copy()
            faiss.normalize_L2(q_norm)
            D, I = self.index.search(q_norm, top_k)
            return I[0], D[0]
        else:
            # normalize query for cosine
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            q_norm = q / norms
            n_neighbors = min(top_k, self.n)
            dists, idxs = self.nn.kneighbors(q_norm, n_neighbors=n_neighbors, return_distance=True)
            sims = 1.0 - dists[0]
            return idxs[0], sims

# 4) Build KB (cache in session_state)
def build_kb(csv_path: str, openai_api_key: Optional[str] = None, sbert_model: str = "all-MiniLM-L6-v2", max_docs: Optional[int] = None) -> Dict[str, Any]:
    docs, metadatas = load_csv_as_docs(csv_path)
    if max_docs:
        docs = docs[:max_docs]
        metadatas = metadatas[:max_docs]
    embedder = Embedder(openai_api_key=openai_api_key, sbert_model_name=sbert_model)
    embeddings = embedder.embed_texts(docs)
    # Use FAISS only if available; otherwise fall back to sklearn
    index = VectorIndex(embeddings, use_faiss=FAISS_AVAILABLE)
    kb = {"docs": docs, "metadatas": metadatas, "embeddings": embeddings, "embedder": embedder, "index": index, "use_faiss": index.use_faiss, "csv_path": csv_path}
    return kb

def retrieve_top_k(kb: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str,Any]]:
    embedder: Embedder = kb["embedder"]
    q_emb = embedder.embed_texts([query])
    idxs, scores = kb["index"].query(q_emb, top_k=top_k)
    results = []
    n_docs = len(kb.get("docs", []))
    for i, sc in zip(idxs, scores):
        # filter out invalid indices (FAISS may return -1 for empty slots)
        try:
            ii = int(i)
        except Exception:
            continue
        if ii < 0 or ii >= n_docs:
            continue
        results.append({"index": ii, "score": float(sc), "doc": kb["docs"][ii], "metadata": kb["metadatas"][ii]})
    return results


# 2b) Relationships helper: compute pairwise correlations among numeric features for the top-K retrieved rows
def compute_relationships(kb: Dict[str, Any], retrieved: List[Dict[str, Any]], top_k: int = 5) -> Optional[Dict[str, Any]]:
    """
    Return correlations and a short summary of relationships for the rows corresponding to retrieved docs.
    Output dict contains:
      - corr_df: DataFrame of pairwise Pearson correlations (as JSON-serializable nested lists)
      - top_pairs: list of (feature_a, feature_b, corr_value) sorted by absolute correlation desc
    If not enough rows or CSV unavailable, returns None.
    """
    if not retrieved:
        return None
    csv_path = kb.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    # collect indices from retrieved
    idxs = [r.get('index') for r in retrieved if r.get('index') is not None]
    idxs = [int(i) for i in idxs if isinstance(i, (int, np.integer, np.int64, np.int32)) or (str(i).isdigit())]
    if not idxs:
        return None
    # ensure indexes in bounds
    idxs = [i for i in idxs if 0 <= i < len(df)]
    if len(idxs) < 2:
        # try to expand to nearby rows (neighborhood) to get more data for correlation
        # pick unique country if available and get all rows for that country
        possible_id_cols = [c for c in df.columns if any(k in c.lower() for k in ["country","pais","region","iso","name"]) ]
        if possible_id_cols:
            first_row = df.iloc[idxs[0]] if idxs else None
            if first_row is not None:
                for col in possible_id_cols:
                    try:
                        val = first_row.get(col)
                        if pd.notna(val):
                            block = df[df[col] == val]
                            if len(block) >= 2:
                                df_sel = block
                                break
                    except Exception:
                        continue
                else:
                    return None
        else:
            return None
    else:
        df_sel = df.iloc[idxs]
    # select numeric columns only
    num_df = df_sel.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2 or num_df.shape[0] < 2:
        return None
    # compute Pearson corr
    corr = num_df.corr(method='pearson')
    # extract top absolute correlations (excluding self)
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for j in range(i+1, len(cols)):
            b = cols[j]
            v = corr.at[a, b]
            if pd.notna(v):
                pairs.append((a, b, float(v)))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs_sorted[:10]
    # return numeric dataframe as well to allow plotting scatter plots
    return {"corr": corr, "top_pairs": top_pairs, "num_df": num_df}

# 5) Generation: local reasoning engine without external API keys.
SYSTEM_PROMPT_TEMPLATE = (
    "Você é um assistente analítico especializado em avaliação de risco geopolítico. "
    "Use apenas as informações fornecidas pelos documentos recuperados e pelo usuário. "
    "Seu output deve conter: (A) um Resumo Executivo curto (2-4 linhas) com uma avaliação do risco imediato; "
    "(B) uma Explicação técnica clara (3-6 linhas) indicando quais sinais/valores suportam essa avaliação; "
    "(C) 3 recomendações práticas alinhadas ao Framework de Ottawa para IA Militar (responsabilidade, supervisão humana, mitigação de riscos). "
    "Formate a resposta com seções separadas e seja conciso."
)

def generate_with_local_reasoner(retrieved_docs: List[Dict[str, Any]], user_query: str, prediction: Optional[Dict[str, Any]] = None, model_prediction: Optional[Dict[str, Any]] = None, relationships: Optional[Dict[str, Any]] = None) -> str:
    """Local reasoning engine that mirrors a compact LLM response without external APIs."""
    combined = "\n".join([r["doc"] for r in retrieved_docs]) if retrieved_docs else ""
    text_lower = combined.lower()

    top_label = prediction.get("top") if prediction else "Escalada Militar"
    top_probs = prediction.get("probs", {}) if prediction else {}
    explanation = prediction.get("explanation", "Sem explicação adicional.") if prediction else "Sem explicação adicional."
    rel_text = ""
    if relationships:
        top_pairs = relationships.get("top_pairs") or []
        if top_pairs:
            rel_text = "\nRelações observadas: " + "; ".join([f"{a} x {b} => r={v:.3f}" for a, b, v in top_pairs[:3]]) + "."

    model_text = ""
    if model_prediction:
        prob = model_prediction.get("probability")
        pred = model_prediction.get("prediction")
        model_text = f"\nProbabilidade do modelo treinado para escalada em 6 meses: {prob:.3f}; rótulo binário: {pred}."
        fi = model_prediction.get("feature_inputs") or {}
        if fi:
            keys = [k for k in ["political_stability_index", "gdp_growth_pct", "military_expenditure_pct_gdp", "border_disputes_count", "refugee_outflow_thousands", "protest_events_last_3m"] if k in fi]
            feature_text = "; ".join([f"{k}={fi[k]}" for k in keys[:4]])
            model_text += f"\nFeatures-chave: {feature_text}."

    escalation_markers = ["attack", "troop", "military", "naval", "escal", "hostility", "strike", "conflict", "sanction"]
    diplomatic_markers = ["agreement", "talk", "dialogue", "negotiation", "summit", "mediation", "diplomat"]
    humanitarian_markers = ["refugee", "displace", "civilian", "humanitarian", "famine", "evacu", "casualti"]
    stable_markers = ["stable", "stability", "status quo", "cooperate", "de-escal", "accord"]

    matched = {
        "escalation": sum(1 for token in escalation_markers if token in text_lower),
        "diplomacy": sum(1 for token in diplomatic_markers if token in text_lower),
        "humanitarian": sum(1 for token in humanitarian_markers if token in text_lower),
        "stability": sum(1 for token in stable_markers if token in text_lower),
    }

    risk_signal = "alta" if matched["escalation"] >= matched["stability"] else "moderada"
    summary = (
        f"Resumo Executivo: a evidência recuperada sustenta um cenário de risco {risk_signal} de escalada geopolítica. "
        f"O desfecho mais provável segundo a projeção heurística é {top_label}."
    )
    if top_probs:
        summary += f" Probabilidades: {top_probs}."

    technical = (
        f"Explicação técnica: os sinais de escalada aparecem em {matched['escalation']} marcadores de conflito, "
        f"enquanto os sinais diplomáticos aparecem em {matched['diplomacy']} e os de estabilidade em {matched['stability']}. "
        f"A descrição da base recuperada e a projeção numérica apontam que as pressões de tensão, mobilização naval e vulnerabilidade ambiental "
        f"estão reforçando risco de deterioração. {explanation}"
    )

    recommendations = (
        "Recomendações (Framework de Ottawa):\n"
        "1) Supervisão Humana: manter revisão humana das decisões e da evidência antes de qualquer ação automatizada.\n"
        "2) Coordenação e Comunicação: priorizar canais diplomáticos e de inteligência regional para reduzir ruído e ambiguidades.\n"
        "3) Mitigação Técnica: reforçar vigilância, coleta adicional e controles de segurança em qualquer automação ou recomendação operacional."
    )

    return f"{SYSTEM_PROMPT_TEMPLATE}\n\n{summary}\n\n{technical}{rel_text}{model_text}\n\n{recommendations}"

def _parse_numeric_params_from_query(query: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    import re
    # look for patterns like Tensão=6.0, NDVI=4.0, Movimentação_Naval=5.0
    tension = ndvi = naval = None
    try:
        m = re.search(r"tensão\s*=\s*([0-9]+\.?[0-9]*)", query, flags=re.IGNORECASE)
        if m:
            tension = float(m.group(1))
        m = re.search(r"ndvi\s*=\s*([0-9]+\.?[0-9]*)", query, flags=re.IGNORECASE)
        if m:
            ndvi = float(m.group(1))
        m = re.search(r"movimentação[_ ]?naval\s*=\s*([0-9]+\.?[0-9]*)", query, flags=re.IGNORECASE)
        if m:
            naval = float(m.group(1))
    except Exception:
        pass
    return tension, ndvi, naval


def predict_outcome_from_evidence(retrieved_docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Heuristic prediction of most likely geopolitical outcome based on retrieved docs and numeric params parsed from query.
    Returns: {top: label, probs: {label:prob}, explanation: str}
    Labels: 'Escalada Militar', 'Estabilidade', 'Resolução Diplomática', 'Crise Humanitária'
    """
    labels = ["Escalada Militar", "Estabilidade", "Resolução Diplomática", "Crise Humanitária"]
    combined = "\n".join([r["doc"] for r in retrieved_docs])
    text_lower = combined.lower()
    # keyword signal weights
    weights = {"escalation": 0.0, "stability": 0.0, "diplomacy": 0.0, "humanitarian": 0.0}
    escal_k = ["attack","troop","military","naval","escal","shelling","bomb","strike","hostilities","invasion"]
    dip_k = ["talk","negotiat","dialogue","agreement","summit","mediate","deal","diplomat","embassy"]
    human_k = ["refugee","displace","casualti","civilian","humanitarian","famine","sanction","crisis","evacuat"]
    stable_k = ["stable","stability","status quo","maintain","low tension","de-escalat","cooperate"]
    for k in escal_k:
        if k in text_lower:
            weights["escalation"] += 1.0
    for k in dip_k:
        if k in text_lower:
            weights["diplomacy"] += 1.0
    for k in human_k:
        if k in text_lower:
            weights["humanitarian"] += 1.0
    for k in stable_k:
        if k in text_lower:
            weights["stability"] += 1.0
    # numeric inputs from query
    tension, ndvi, naval = _parse_numeric_params_from_query(query)
    numeric_score = 0.0
    if tension is not None:
        numeric_score += (tension / 10.0) * 2.0
    if naval is not None:
        numeric_score += (naval / 10.0) * 1.5
    if ndvi is not None:
        # lower NDVI (higher anomaly) increases risk
        numeric_score += ((10.0 - ndvi) / 10.0) * 1.0
    # combine weights and numeric_score
    esc = weights["escalation"] * 1.2 + numeric_score
    dip = weights["diplomacy"] * 1.0
    hum = weights["humanitarian"] * 1.1 + max(0.0, numeric_score - 1.5)
    sta = weights["stability"] * 0.8 + max(0.0, 3.0 - numeric_score)
    raw = [esc, sta, dip, hum]
    # softmax to probabilities
    import math
    exps = [math.exp(r) for r in raw]
    s = sum(exps) if sum(exps) > 0 else 1.0
    probs = [float(e / s) for e in exps]
    probs_dict = {lab: round(float(p), 3) for lab, p in zip(labels, probs)}
    top_idx = int(np.argmax(probs))
    explanation = f"Sinais textuais: escalation={weights['escalation']}, diplomacy={weights['diplomacy']}, humanitarian={weights['humanitarian']}, stability={weights['stability']}.\n"
    explanation += f"Score numérico agregado (Tensão/Naval/NDVI): {numeric_score:.2f}.\n"
    explanation += f"Cálculo combinado conduz ao desfecho mais provável: {labels[top_idx]} ({probs_dict[labels[top_idx]]*100:.1f}%)."

    # Parameter contribution breakdown (normalize contributions to sum to 1 for interpretability)
    contrib_escalation = weights['escalation'] * 1.2
    contrib_diplomacy = weights['diplomacy'] * 1.0
    contrib_humanitarian = weights['humanitarian'] * 1.1
    contrib_stability = weights['stability'] * 0.8
    contrib_numeric = numeric_score
    contrib_list = [abs(contrib_escalation), abs(contrib_stability), abs(contrib_diplomacy), abs(contrib_humanitarian), abs(contrib_numeric)]
    total_contrib = sum(contrib_list) if sum(contrib_list) > 0 else 1.0
    parameter_contributions = {
        'escalation_keywords': round(float(contrib_escalation)/total_contrib, 3),
        'stability_keywords': round(float(contrib_stability)/total_contrib, 3),
        'diplomacy_keywords': round(float(contrib_diplomacy)/total_contrib, 3),
        'humanitarian_keywords': round(float(contrib_humanitarian)/total_contrib, 3),
        'numeric_parameters': round(float(contrib_numeric)/total_contrib, 3)
    }

    # Simple future projection: simulate small perturbations and recompute softmax to show sensitivity
    def _simulate(t_inc=0.0, n_inc=0.0, ndvi_delta=0.0):
        t2 = tension + t_inc if tension is not None else None
        n2 = naval + n_inc if naval is not None else None
        nd2 = ndvi + ndvi_delta if ndvi is not None else None
        ns = 0.0
        if t2 is not None:
            ns += (t2/10.0) * 2.0
        if n2 is not None:
            ns += (n2/10.0) * 1.5
        if nd2 is not None:
            ns += ((10.0 - nd2) / 10.0) * 1.0
        esc2 = weights['escalation'] * 1.2 + ns
        dip2 = weights['diplomacy'] * 1.0
        hum2 = weights['humanitarian'] * 1.1 + max(0.0, ns - 1.5)
        sta2 = weights['stability'] * 0.8 + max(0.0, 3.0 - ns)
        exps2 = [math.exp(x) for x in [esc2, sta2, dip2, hum2]]
        s2 = sum(exps2) if sum(exps2) > 0 else 1.0
        probs2 = [float(e/s2) for e in exps2]
        return {lab: round(float(p),3) for lab,p in zip(labels, probs2)}

    # baseline already in probs_dict; create stress scenarios
    projection = {
        'baseline': probs_dict,
        'tension_plus_1': _simulate(t_inc=1.0),
        'naval_plus_1': _simulate(n_inc=1.0),
        'worse_ndvi_minus_1': _simulate(ndvi_delta=-1.0),
        'combined_worse': _simulate(t_inc=1.0, n_inc=1.0, ndvi_delta=-1.0)
    }

    # human-friendly explanation of contributions and projection
    contrib_text = (
        "Contribuições dos parâmetros (normalizadas): " + ", ".join([f"{k}={v:.2f}" for k,v in parameter_contributions.items()]) + ".\n"
    )
    proj_text = "Projeção rápida: alterar Tensão/Naval/NDVI em +1/+1/-1 muda as probabilidades para (combined_worse): " + ", ".join([f"{k}={v*100:.1f}%" for k,v in projection['combined_worse'].items()]) + "."

    # extend explanation
    explanation += "\n" + contrib_text + "\n" + proj_text

    return {"top": labels[top_idx], "probs": probs_dict, "explanation": explanation, "parsed_params": {"tension": tension, "ndvi": ndvi, "naval": naval}, 'parameter_contributions': parameter_contributions, 'projection': projection}


def heuristic_generate(retrieved_docs: List[Dict[str, Any]], user_query: str) -> str:
    """Local heuristic generator that preserves the existing projection logic, but uses a local reasoner instead of external LLM APIs."""
    try:
        pred = predict_outcome_from_evidence(retrieved_docs, user_query)
    except Exception:
        pred = None
    return generate_with_local_reasoner(retrieved_docs, user_query, prediction=pred)


def _load_trained_model(model_path: Optional[str] = None):
    import joblib
    if model_path is None:
        model_path = os.path.expanduser('~/Downloads/kb_artifacts/escalation_model_pipeline.joblib')
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        return None


def _model_predict_from_retrieved(kb: Dict[str, Any], retrieved: List[Dict[str, Any]], model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Use the trained pipeline to predict escalation for the top retrieved document's CSV row."""
    model = _load_trained_model(model_path)
    if model is None:
        return None
    csv_path = kb.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return None
    # pick the top retrieved doc (first)
    if not retrieved:
        return None
    top = retrieved[0]
    meta = top.get('metadata', {})
    idx = meta.get('index') if meta else None
    if idx is None:
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        # use positional iloc based on index
        row = df.iloc[int(idx)].copy()
    except Exception:
        return None
    # Build feature dict expected by the pipeline (same features used in training)
    feature_cols = ['political_stability_index','gdp_growth_pct','inflation_rate','unemployment_rate','food_price_index','energy_dependency_pct','military_expenditure_pct_gdp','arms_imports_index','border_disputes_count','refugee_outflow_thousands','sanctions_active','media_freedom_score','protest_events_last_3m','cyber_attack_incidents','trade_dependency_rival_pct','foreign_troops_present','social_media_sentiment','rolling_protest_avg_6m','region','regime_type','month']
    data = {}
    # prepare month parsing
    m = str(row.get('month', ''))
    try:
        year = int(m.split('-')[0]) if '-' in m else int(m)
    except Exception:
        year = 0
    try:
        mon = int(m.split('-')[1]) if '-' in m else 1
    except Exception:
        mon = 1
    data['month_year'] = year
    data['month_mon'] = mon
    for c in feature_cols:
        if c == 'month':
            continue
        data[c] = row.get(c)
    # create DataFrame with single row
    X_row = pd.DataFrame([data])
    # Some pipelines expect exact column order/columns; attempt predict
    try:
        proba = float(model.predict_proba(X_row)[:,1][0])
        pred = int(model.predict(X_row)[0])
        # include feature inputs used for transparency in the UI/report
        feature_inputs = X_row.to_dict(orient='records')[0]
        return {'probability': proba, 'prediction': pred, 'feature_inputs': feature_inputs}
    except Exception:
        # last resort: try to align columns if pipeline has preprocessing
        try:
            proba = float(model.predict_proba(X_row)[0][1])
            pred = int(model.predict(X_row)[0])
            feature_inputs = X_row.to_dict(orient='records')[0]
            return {'probability': proba, 'prediction': pred, 'feature_inputs': feature_inputs}
        except Exception:
            return None


def answer_query_using_kb(kb: Dict[str,Any], query: str, openai_api_key: Optional[str] = None, top_k: int = 5) -> Dict[str,Any]:
    # openai_api_key is ignored intentionally: this app is designed to work without external LLM keys.
    retrieved = retrieve_top_k(kb, query, top_k=top_k)
    try:
        prediction = predict_outcome_from_evidence(retrieved, query)
    except Exception:
        prediction = None
    try:
        model_pred = _model_predict_from_retrieved(kb, retrieved)
    except Exception:
        model_pred = None
    try:
        relationships = compute_relationships(kb, retrieved, top_k=top_k)
    except Exception:
        relationships = None
    ans = generate_with_local_reasoner(retrieved, query, prediction=prediction, model_prediction=model_pred, relationships=relationships)
    return {"answer": ans, "retrieved": retrieved, "mode": "local_reasoner", "prediction": prediction, "model_prediction": model_pred, "relationships": relationships}

# -----------------------
# Streamlit UI
# -----------------------
if st is not None:
    if 'kb' not in st.session_state:
        st.session_state.kb = None
        st.session_state.kb_info = {"built": False, "n_docs": 0, "time": None, "mode": None}

    default_csv_path = os.path.join(os.getcwd(), 'geopolitical_conflict_risk_dataset.csv')
    if st.session_state.kb is None and os.path.exists(default_csv_path):
        try:
            st.session_state.kb = build_kb(default_csv_path, openai_api_key=None, sbert_model='all-MiniLM-L6-v2', max_docs=0)
            st.session_state.kb_info = {"built": True, "n_docs": len(st.session_state.kb['docs']), "time": datetime.now().isoformat(), "mode": 'local_reasoner'}
        except Exception:
            st.session_state.kb = None

    def get_kb_dataframe(kb: Dict[str, Any]) -> Optional[pd.DataFrame]:
        csv_path = kb.get('csv_path') if isinstance(kb, dict) else None
        if not csv_path or not os.path.exists(csv_path):
            return None
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception:
            return None

    def get_country_options(kb: Dict[str, Any]) -> List[str]:
        df = get_kb_dataframe(kb)
        if df is None or 'country' not in df.columns:
            return []
        vals = [str(x) for x in df['country'].dropna().unique().tolist()]
        return sorted(vals)

    EXPECTED_GEO_COLUMNS = [
        'country','region','month','political_stability_index','gdp_growth_pct','inflation_rate','unemployment_rate',
        'food_price_index','energy_dependency_pct','military_expenditure_pct_gdp','arms_imports_index','border_disputes_count',
        'refugee_outflow_thousands','sanctions_active','media_freedom_score','protest_events_last_3m','cyber_attack_incidents',
        'last_conflict_year','trade_dependency_rival_pct','foreign_troops_present','election_cycle','regime_type','social_media_sentiment',
        'rolling_protest_avg_6m','instability_score','conflict_escalation_6m'
    ]

    def validate_geopolitical_csv(df: pd.DataFrame) -> List[str]:
        if df is None or df.empty:
            return ['CSV vazio.']
        missing = [c for c in EXPECTED_GEO_COLUMNS if c not in df.columns]
        return missing

    def get_country_options_from_df(df: pd.DataFrame) -> List[str]:
        if df is None or 'country' not in df.columns:
            return []
        options = [str(x) for x in df['country'].dropna().unique().tolist()]
        return sorted(dict.fromkeys(options))

    def _safe_scale(values: pd.Series, low: float = 0.0, high: float = 10.0) -> float:
        s = pd.to_numeric(values, errors='coerce').dropna()
        if s.empty:
            return 5.0
        lo = float(s.min())
        hi = float(s.max())
        if hi == lo:
            return 5.0
        return float(np.clip((float(s.mean()) - lo) / (hi - lo) * (high - low) + low, low, high))

    def derive_country_context_metrics(kb: Dict[str, Any], countries: List[str]) -> Dict[str, Any]:
        df = get_kb_dataframe(kb)
        base = {"tension": 5.0, "ndvi": 5.0, "naval": 5.0, "summary": "Sem contexto de país definido.", "per_country": [], "risk_leader": None}
        if df is None or not countries or 'country' not in df.columns:
            return base
        subset = df[df['country'].isin(countries)].copy()
        if subset.empty:
            return base
        if 'month' in subset.columns:
            subset['month'] = pd.to_datetime(subset['month'], errors='coerce')
            subset = subset.dropna(subset=['month'])
        if subset.empty:
            return base
        latest = subset.sort_values('month').groupby('country', as_index=False).tail(1)
        latest['risk_score'] = 0.0
        for c in ['border_disputes_count', 'protest_events_last_3m', 'cyber_attack_incidents', 'conflict_escalation_6m', 'refugee_outflow_thousands']:
            if c in latest.columns:
                latest['risk_score'] += pd.to_numeric(latest[c], errors='coerce').fillna(0)
        for c in ['political_stability_index', 'gdp_growth_pct', 'media_freedom_score']:
            if c in latest.columns:
                latest['risk_score'] += (10.0 - pd.to_numeric(latest[c], errors='coerce').fillna(5)) * 0.5

        if 'risk_score' in latest.columns:
            leader = latest.sort_values('risk_score', ascending=False).iloc[0]
            risk_leader = {"country": str(leader.get('country')), "risk_score": float(leader.get('risk_score', 0.0))}
        else:
            risk_leader = None

        tension_cols = [c for c in ['border_disputes_count', 'protest_events_last_3m', 'cyber_attack_incidents', 'conflict_escalation_6m'] if c in latest.columns]
        naval_cols = [c for c in ['military_expenditure_pct_gdp', 'arms_imports_index', 'foreign_troops_present'] if c in latest.columns]
        ndvi_cols = [c for c in ['political_stability_index', 'energy_dependency_pct', 'refugee_outflow_thousands', 'gdp_growth_pct'] if c in latest.columns]

        tension = _safe_scale(latest[tension_cols].stack()) if tension_cols else 5.0
        naval = _safe_scale(latest[naval_cols].stack()) if naval_cols else 5.0
        ndvi = _safe_scale(latest[ndvi_cols].stack()) if ndvi_cols else 5.0
        ndvi = float(np.clip(10.0 - ndvi, 0.0, 10.0)) if ndvi_cols else 5.0

        summary_cols = []
        for c in ['country', 'region', 'month', 'political_stability_index', 'military_expenditure_pct_gdp', 'border_disputes_count', 'protest_events_last_3m', 'cyber_attack_incidents', 'conflict_escalation_6m']:
            if c in latest.columns:
                summary_cols.append(c)
        summary_rows = []
        for _, row in latest[summary_cols].head(3).iterrows():
            row_text = f"{row.get('country', 'País')}"
            if 'month' in row and pd.notna(row['month']):
                row_text += f" ({row['month']})"
            if 'conflict_escalation_6m' in row and pd.notna(row['conflict_escalation_6m']):
                row_text += f" | escalada={float(row['conflict_escalation_6m']):.2f}"
            summary_rows.append(row_text)
        summary = 'Contexto derivado dos polos selecionados: ' + '; '.join(summary_rows) if summary_rows else 'Sem observações recentes para os polos selecionados.'

        per_country = []
        for _, row in latest.iterrows():
            entry = {"country": str(row.get('country', ''))}
            for c in ['political_stability_index', 'military_expenditure_pct_gdp', 'border_disputes_count', 'refugee_outflow_thousands', 'protest_events_last_3m', 'cyber_attack_incidents', 'conflict_escalation_6m']:
                if c in row and pd.notna(row[c]):
                    entry[c] = float(row[c])
            per_country.append(entry)

        return {
            "tension": float(np.clip(tension, 0.0, 10.0)),
            "ndvi": float(np.clip(ndvi, 0.0, 10.0)),
            "naval": float(np.clip(naval, 0.0, 10.0)),
            "summary": summary,
            "per_country": per_country,
            "risk_leader": risk_leader,
        }

    csv_file = None
    with st.sidebar:
        st.header("Contexto do Conflito")
        st.markdown("Base de Conhecimento (CSV)")
        csv_file = st.file_uploader("Anexe o CSV base do conflito (country-month rows)", type=["csv"])
        scenario = st.selectbox("Cenário Padrão", options=[
            "Disputas no Atlântico Sul (Pré-sal x ZOPACAS)",
            "Crise Fronteiriça - Região Norte",
            "Crise Diplomática - Trafego Marítimo"
        ], index=0)
        st.markdown("### Polos de tensão do conflito")
        country_options = []
        preview_df = None
        if csv_file is not None:
            try:
                preview_df = pd.read_csv(io.BytesIO(csv_file.getvalue()), low_memory=False)
                missing = validate_geopolitical_csv(preview_df)
                if missing:
                    st.warning("CSV anexado fora do schema esperado. Colunas faltantes: " + ", ".join(missing))
                else:
                    country_options = get_country_options_from_df(preview_df)
            except Exception as exc:
                st.warning(f"Não foi possível ler o CSV anexado: {exc}")
                country_options = []
        if not country_options and st.session_state.kb is not None:
            country_options = get_country_options(st.session_state.kb)
        if not country_options and os.path.exists(default_csv_path):
            try:
                preview_df = pd.read_csv(default_csv_path, low_memory=False)
                country_options = get_country_options_from_df(preview_df)
            except Exception:
                country_options = []

        default_p1 = country_options[:1] if country_options else []
        default_p2 = country_options[1:2] if len(country_options) > 1 else []
        pole_1 = st.multiselect(
            "Polo 1 — ator principal / país de origem",
            options=country_options,
            default=default_p1,
            help="Selecione o primeiro polo da tensão (país ou ator principal)."
        )
        pole_2 = st.multiselect(
            "Polo 2 — ator rival / contraparte",
            options=country_options,
            default=default_p2,
            help="Selecione o segundo polo da tensão (país ou ator rival)."
        )
        if len(pole_1) == 0 or len(pole_2) == 0:
            st.warning("Selecione pelo menos um país em cada polo para que o risco seja calculado.")
        selected_countries = list(dict.fromkeys(pole_1 + pole_2))
        st.caption("Parâmetros não editáveis: a análise usa os dados reais e os indicadores associados aos polos selecionados.")
        st.markdown("---")
        st.markdown("Motor local de análise")
        st.caption("Este app usa o mecanismo local de projeção e raciocínio integrado, sem chaves externas de LLM.")

        if csv_file is not None:
            st.caption(f"CSV carregado: {csv_file.name}")
        st.markdown("---")
        st.markdown("Opções avançadas")
        max_docs = st.number_input("Max docs a indexar (0 = todos)", min_value=0, max_value=10000, value=0, step=1)
        build_kb_btn = st.button("Construir / Recarregar KB a partir do CSV")

    tabs = st.tabs(["Painel de Simulação", "Relatório Estratégico", "KB / Debug"])

    # Prepare inputs: no manual editing; values are derived from the selected countries/context.
    inputs = {"tension": 5.0, "ndvi": 5.0, "naval": 5.0}
    if st.session_state.kb is not None and selected_countries:
        derived = derive_country_context_metrics(st.session_state.kb, selected_countries)
        inputs = {"tension": derived["tension"], "ndvi": derived["ndvi"], "naval": derived["naval"]}
        st.session_state['last_country_context'] = derived
    df_sim = simulate_evolution(inputs["tension"], inputs["ndvi"], inputs["naval"], steps=12, seed=42)
    fig = create_3d_plot(df_sim)
    sentiment_label, sentiment_color = analyze_sentiment_geopolitical(inputs["tension"])

    # KB build / caching in session_state
    if "kb" not in st.session_state:
        st.session_state.kb = None
        st.session_state.kb_info = {"built": False, "n_docs": 0, "time": None, "mode": None}

    def build_kb_from_upload(uploaded_file, key: Optional[str] = None, max_docs_param: int = 0):
        if uploaded_file is None:
            st.warning("Nenhum CSV fornecido. Faça upload de um CSV para construir a base de conhecimento.")
            return None
        # save to temp path
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp_path = tmp.name
            try:
                tmp.write(uploaded_file.getbuffer())
                tmp.flush()
            finally:
                tmp.close()
            st.info(f"Lendo CSV ({tmp_path}) e construindo KB... Isso pode demorar alguns instantes.")
            kb = build_kb(tmp_path, openai_api_key=None, sbert_model="all-MiniLM-L6-v2", max_docs=(None if max_docs_param==0 else max_docs_param))
            st.success("KB construída com sucesso.")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return kb
        except Exception as e:
            st.error(f"Erro ao construir KB: {e}")
            return None

    # If user clicked build or session has no kb and a file uploaded, build it
    if build_kb_btn:
        st.session_state.kb = build_kb_from_upload(csv_file, None, int(max_docs))
        if st.session_state.kb:
            st.session_state.kb_info = {"built": True, "n_docs": len(st.session_state.kb["docs"]), "time": datetime.now().isoformat(), "mode": "local_reasoner"}

    # If no explicit build but file present and kb not built, auto-build (light)
    if st.session_state.kb is None and csv_file is not None:
        # lazy auto-build small preview: do not auto-build huge sets without confirmation
        if st.button("Auto-construir KB (rápido)"):
            st.session_state.kb = build_kb_from_upload(csv_file, None, int(max_docs))
            if st.session_state.kb:
                st.session_state.kb_info = {"built": True, "n_docs": len(st.session_state.kb["docs"]), "time": datetime.now().isoformat(), "mode": "local_reasoner"}

    # Use the RAG system to answer a query derived from sliders when requested
    def query_from_sliders(tension_v, ndvi_v, naval_v):
        return f"Dado o cenário com Tensão={tension_v:.1f}, NDVI={ndvi_v:.1f}, Movimentação_Naval={naval_v:.1f}, avalie o risco de escalada nos próximos 6 meses, explique as razões e recomende ações alinhadas ao Framework de Ottawa."

    def build_country_context_chart(kb: Dict[str, Any], countries: List[str]) -> Optional[go.Figure]:
        df = get_kb_dataframe(kb)
        if df is None or 'country' not in df.columns or not countries:
            return None
        subset = df[df['country'].isin(countries)].copy()
        if subset.empty:
            return None
        relevant = [
            'month', 'country', 'political_stability_index', 'military_expenditure_pct_gdp',
            'border_disputes_count', 'refugee_outflow_thousands', 'protest_events_last_3m',
            'cyber_attack_incidents', 'conflict_escalation_6m'
        ]
        missing = [c for c in relevant if c not in subset.columns]
        if missing:
            relevant = [c for c in relevant if c in subset.columns]
        if len(relevant) < 3:
            return None
        subset['month'] = pd.to_datetime(subset['month'], errors='coerce')
        subset = subset.dropna(subset=['month'])
        if subset.empty:
            return None
        melt = subset.melt(id_vars=['country', 'month'], value_vars=[c for c in relevant if c not in ['country','month']], var_name='indicador', value_name='valor')
        fig = px.line(
            melt,
            x='month',
            y='valor',
            color='country',
            line_group='indicador',
            facet_col='indicador',
            facet_col_wrap=2,
            title='Indicadores relevantes para agravamento ou contenção do conflito'
        )
        fig.update_layout(height=700, template='plotly_white')
        return fig

    # Tab 0: Painel de Simulação
    with tabs[0]:
        st.header("Painel de Simulação")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Proxy 4D (visualização 3D interativa)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("A linha mostra a trajetória temporal simulada. Ajuste sliders para atualizar.")
        with col2:
            st.subheader("Resumo Rápido")
            current_risk = compute_risk(inputs["tension"], inputs["ndvi"], inputs["naval"])
            st.metric("Risco de Escalada (estimado)", f"{current_risk:.1f} %")
            st.markdown("**Sentimento do Discurso Diplomático**")
            badge_html = f"""<div style="background-color:{sentiment_color};padding:10px;border-radius:6px;color:white;text-align:center;"><strong>{sentiment_label}</strong></div>"""
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("---")
            if st.session_state.kb is not None and selected_countries:
                derived = st.session_state.get('last_country_context', {})
                if derived.get('risk_leader'):
                    leader = derived['risk_leader']
                    st.subheader("Líder do risco no contexto selecionado")
                    st.write(f"- País/ator com maior contribuição ao risco: {leader['country']} (score {leader['risk_score']:.2f})")
                per_country = derived.get('per_country', [])
                if per_country:
                    st.subheader("Comparativo por país/ator")
                    df_comp = pd.DataFrame(per_country)
                    st.dataframe(df_comp, use_container_width=True)
                context_fig = build_country_context_chart(st.session_state.kb, selected_countries)
                if context_fig is not None:
                    st.subheader("Indicadores relevantes por país/ator")
                    st.plotly_chart(context_fig, use_container_width=True)
                    st.caption("Evolução mensal dos principais indicadores: estabilidade política, gasto militar, disputas fronteiriças, protestos, refúgio e escalada de conflito.")
            st.markdown("---")
            st.subheader("Análise RAG (CSV -> KB -> Resposta)")
            if st.session_state.kb is None:
                st.info("Nenhuma KB construída. Faça upload de um CSV na barra lateral e clique em 'Construir KB'.")
            else:
                if st.button("Gerar Análise (RAG) a partir dos sliders"):
                    user_q = query_from_sliders(inputs["tension"], inputs["ndvi"], inputs["naval"])
                    with st.spinner("Recuperando evidências e gerando resposta..."):
                        result = answer_query_using_kb(st.session_state.kb, user_q, top_k=5)
                    # store last result for report transparency
                    try:
                        st.session_state['last_rag_result'] = result
                    except Exception:
                        pass
                    st.subheader("Resposta (RAG)")
                    st.write(result.get("answer", "(sem resposta)"))
                    st.markdown(f"Modo: {result.get('mode')}")
                    pred = result.get('prediction')
                    # show heuristic prediction summary if available
                    if pred:
                        st.subheader("Predição Heurística (RAG)")
                        st.write(f"Desfecho provável: {pred['top']} — probabilidades: {pred['probs']}")
                        st.write(pred['explanation'])
                    # show parameter contributions and projection
                    pc = pred.get('parameter_contributions') if pred else None
                    if pc:
                        st.markdown('**Contribuições dos parâmetros (normalizadas):**')
                        for k,v in pc.items():
                            st.write(f"- {k}: {v}")
                    proj = pred.get('projection')
                    if proj and proj.get('combined_worse'):
                        cw = proj.get('combined_worse')
                        st.markdown('**Projeção (cenário piorado - Tensão+1, Naval+1, NDVI-1):**')
                        for k,v in cw.items():
                            st.write(f"- {k}: {v*100:.1f}%")
                    # show trained model prediction if available
                    if result.get('model_prediction'):
                        mp = result['model_prediction']
                        st.subheader("Predição do Modelo Treinado")
                        st.write(f"Probabilidade de escalada (6m): {mp.get('probability'):.3f} — Predição: {mp.get('prediction')}")
                        if 'feature_inputs' in mp:
                            st.markdown("**Parâmetros do modelo usados na predição (resumidos):**")
                            fi = mp.get('feature_inputs')
                            # show only a selection of key features for readability
                            keys_show = [k for k in ['political_stability_index','gdp_growth_pct','military_expenditure_pct_gdp','border_disputes_count','refugee_outflow_thousands','protest_events_last_3m'] if k in fi]
                            for k in keys_show:
                                st.write(f"- {k}: {fi.get(k)}")
                    # free text question in the simulation panel
                st.markdown("---")
                st.subheader("Pergunta livre (texto)")
                q_free = st.text_area("Faça uma pergunta livre à KB (ex.: 'Qual é o risco de escalada para este país no próximo semestre?')", key='q_sim', height=100)
                qcols = st.columns(2)
                if qcols[0].button("Perguntar (texto)"):
                    if q_free.strip() == '':
                        st.warning('Digite uma pergunta antes de enviar.')
                    else:
                        with st.spinner('Recuperando evidências e gerando resposta...'):
                            result = answer_query_using_kb(st.session_state.kb, q_free, top_k=5)
                        try:
                            st.session_state['last_rag_result'] = result
                        except Exception:
                            pass
                        st.subheader('Resposta (RAG)')
                        st.write(result.get('answer', '(sem resposta)'))
                        st.markdown(f"Modo: {result.get('mode')}")
                        pred = result.get('prediction')
                        if pred:
                            st.subheader('Predição Heurística (RAG)')
                            st.write(f"Desfecho provável: {pred['top']} — probabilidades: {pred['probs']}")
                            st.write(pred['explanation'])
                            pc = pred.get('parameter_contributions')
                            if pc:
                                st.markdown('**Contribuições dos parâmetros (normalizadas):**')
                                for k,v in pc.items():
                                    st.write(f"- {k}: {v}")
                            proj = pred.get('projection')
                            if proj and proj.get('combined_worse'):
                                cw = proj.get('combined_worse')
                                st.markdown('**Projeção (cenário piorado - Tensão+1, Naval+1, NDVI-1):**')
                                for k,v in cw.items():
                                    st.write(f"- {k}: {v*100:.1f}%")
                        if result.get('model_prediction'):
                            mp = result['model_prediction']
                            st.subheader('Predição do Modelo Treinado')
                            st.write(f"Probabilidade de escalada (6m): {mp.get('probability'):.3f} — Predição: {mp.get('prediction')}")
                            if 'feature_inputs' in mp:
                                st.markdown('**Parâmetros do modelo usados na predição (resumidos):**')
                                fi = mp.get('feature_inputs')
                                keys_show = [k for k in ['political_stability_index','gdp_growth_pct','military_expenditure_pct_gdp','border_disputes_count','refugee_outflow_thousands','protest_events_last_3m'] if k in fi]
                                for k in keys_show:
                                    st.write(f"- {k}: {fi.get(k)}")
                        # show relationships (correlations) if available
                        rel = result.get('relationships')
                        if rel:
                            corr = rel.get('corr')
                            top_pairs = rel.get('top_pairs', [])
                            try:
                                st.markdown('**Relações observadas entre os parâmetros (top-K):**')
                                # show heatmap if plotly available
                                if 'corr' in rel and corr is not None and isinstance(corr, pd.DataFrame):
                                    fig_rel = px.imshow(corr, text_auto=True, aspect='auto', title='Matriz de Correlação (top-K)')
                                    st.plotly_chart(fig_rel, use_container_width=True)
                                # show top correlated pairs
                                if top_pairs:
                                    st.markdown('**Top pares correlacionados (|r| mais alta):**')
                                    for a,b,v in top_pairs:
                                        st.write(f"- {a} <-> {b}: r={v:.3f}")
                                # generate scatter plots for top 3 pairs and save in session_state for report
                                try:
                                    imgs = st.session_state.get('last_rag_images', {})
                                    num_df = rel.get('num_df')
                                    saved_names = []
                                    for i, (a,b,v) in enumerate(top_pairs[:3]):
                                        if num_df is None or a not in num_df.columns or b not in num_df.columns:
                                            continue
                                        # dropna
                                        sel = num_df[[a,b]].dropna()
                                        if sel.shape[0] < 2:
                                            continue
                                        fig_scatter = px.scatter(sel, x=a, y=b, title=f'{a} vs {b} (r={v:.3f})', trendline=None)
                                        # add linear fit line
                                        try:
                                            xs = sel[a].to_numpy()
                                            ys = sel[b].to_numpy()
                                            coeffs = np.polyfit(xs, ys, deg=1)
                                            x_line = np.linspace(xs.min(), xs.max(), 50)
                                            y_line = np.polyval(coeffs, x_line)
                                            fig_scatter.add_traces(px.line(x=x_line, y=y_line, name='fit').data)
                                        except Exception:
                                            pass
                                        st.plotly_chart(fig_scatter, use_container_width=True)
                                        # save as html and png (png optional)
                                        try:
                                            html_str = fig_scatter.to_html(full_html=False)
                                            name_html = f'rel_scatter_{i}_{a}_{b}.html'
                                            imgs[name_html] = html_str.encode('utf-8')
                                            # try png
                                            try:
                                                png_bytes = fig_scatter.to_image(format='png')
                                                name_png = f'rel_scatter_{i}_{a}_{b}.png'
                                                imgs[name_png] = png_bytes
                                                saved_names.append(name_png)
                                            except Exception:
                                                # png not available
                                                saved_names.append(name_html)
                                        except Exception:
                                            pass
                                    if imgs:
                                        st.session_state['last_rag_images'] = imgs
                                except Exception:
                                    pass
                            except Exception:
                                # fallback textual
                                if top_pairs:
                                    st.write('Relações (top pares):')
                                    for a,b,v in top_pairs:
                                        st.write(f"- {a} <-> {b}: r={v:.3f}")
                if qcols[1].button('Limpar pergunta'):
                    try:
                        st.session_state['q_sim'] = ''
                    except Exception:
                        pass
                    st.subheader("Evidências recuperadas")
                    for r in result["retrieved"]:
                        st.markdown(f"- score={r['score']:.3f} — {r['doc'][:400]}...")
            st.markdown("---")
            st.info("Observação: o app usa apenas o motor local de projeção e raciocínio interno, sem dependência de chave externa de LLM.")

    # Tab 1: Relatório Estratégico
    with tabs[1]:
        st.header("Relatório Estratégico")
        st.markdown("Se uma KB foi construída, gere a Análise RAG no Painel de Simulação e baixe o relatório aqui.")
        if st.session_state.kb:
            if st.button("Gerar relatório TXT com a resposta atual (use botão de análise primeiro)"):
                # Prefer using last stored RAG result for transparency; fall back to recomputing
                result = st.session_state.get('last_rag_result')
                if result is None:
                    user_q = query_from_sliders(inputs["tension"], inputs["ndvi"], inputs["naval"])
                    result = answer_query_using_kb(st.session_state.kb, user_q, top_k=5)
                # Build report with explicit Parameters Analyzed section
                report = f"Relatório Estratégico - {scenario}\nGerado em: {datetime.now().isoformat()}\n\nParâmetros do Cenário:\n- Tensão: {inputs['tension']}\n- NDVI: {inputs['ndvi']}\n- Movimentação Naval: {inputs['naval']}\n\nResposta RAG:\n{result.get('answer', '(sem resposta)')}\n\nModo: {result.get('mode')}\n\nParâmetros Analisados e Evidências:\n"
                # include parsed numeric params from heuristic prediction (if available)
                pred = result.get('prediction')
                if pred:
                    parsed = pred.get('parsed_params', {})
                    report += "- Parâmetros extraídos da pergunta:\n"
                    report += f"  - Tensão (se fornecida): {parsed.get('tension')}\n"
                    report += f"  - NDVI (se fornecida): {parsed.get('ndvi')}\n"
                    report += f"  - Movimentação Naval (se fornecida): {parsed.get('naval')}\n"
                report += f"- Heurística de predição: {pred.get('explanation')}\n"
                # include parameter contributions and simple projection
                pc = pred.get('parameter_contributions')
                if pc:
                    report += "- Contribuições dos parâmetros (normalizadas):\n"
                    for k,v in pc.items():
                        report += f"  - {k}: {v}\n"
                proj = pred.get('projection')
                if proj and proj.get('combined_worse'):
                    cw = proj.get('combined_worse')
                    report += "- Projeção (cenário piorado - Tensão+1, Naval+1, NDVI-1):\n"
                    for k,v in cw.items():
                        report += f"  - {k}: {v*100:.1f}%\n"
                report += "\n"
                # include model prediction and features if available
                mp = result.get('model_prediction')
                if mp:
                    report += f"- Predição do modelo treinado (probabilidade escalada 6m): {mp.get('probability')} (predição: {mp.get('prediction')})\n"
                    fi = mp.get('feature_inputs')
                    if fi:
                        report += "- Principais features usadas no modelo:\n"
                        for k in ['political_stability_index','gdp_growth_pct','military_expenditure_pct_gdp','border_disputes_count','refugee_outflow_thousands','protest_events_last_3m']:
                            if k in fi:
                                report += f"  - {k}: {fi.get(k)}\n"
                    report += "\n"
                # list retrieved docs with scores and indices
                report += "Evidências recuperadas (top-K):\n"
                # add clear list of source bases (which CSV rows drove the conclusion)
                try:
                    csv_path_used = st.session_state.kb.get('csv_path') if st.session_state.kb else None
                except Exception:
                    csv_path_used = None
                report += "\nBases que direcionam a conclusão (linhas/entradas do CSV):\n"
                # include relationship summary if available
                rel = result.get('relationships')
                if rel:
                    top_pairs = rel.get('top_pairs', [])
                    if top_pairs:
                        report += "\nRelações observadas entre parâmetros (top pares correlacionados):\n"
                        for a,b,v in top_pairs:
                            report += f"- {a} <-> {b}: r={v:.3f}\n"
                        report += "\n"
                seen_idxs = set()
                df_src = None
                id_cols = []
                if csv_path_used and os.path.exists(csv_path_used):
                    try:
                        df_src = pd.read_csv(csv_path_used, low_memory=False)
                        id_cols = [c for c in df_src.columns if any(k in c.lower() for k in ["country","pais","region","iso","name","id","month","country_name","country_code"])][:6]
                    except Exception:
                        df_src = None
                        id_cols = []
                for r in result.get('retrieved', []):
                    idx = r.get('index')
                    if idx in seen_idxs:
                        continue
                    seen_idxs.add(idx)
                    if df_src is not None and isinstance(idx, int) and 0 <= int(idx) < len(df_src):
                        row = df_src.iloc[int(idx)]
                        id_parts = []
                        for c in id_cols:
                            try:
                                if c in df_src.columns and pd.notna(row[c]):
                                    id_parts.append(f"{c}={row[c]}")
                            except Exception:
                                continue
                        id_str = "; ".join(id_parts) if id_parts else "(sem identificadores claros)"
                        report += f"- CSV: {csv_path_used} | row_index={idx} | {id_str}\n"
                    else:
                        report += f"- row_index={idx} (Fonte não disponível)\n"
                report += "\nEvidências detalhadas (top-K):\n"
                for r in result.get('retrieved', []):
                    report += f"- idx={r.get('index')} score={r.get('score'):.3f}: {r.get('doc')[:800]}\n\n"
                # always offer the plain TXT
                st.download_button("Baixar Relatório (TXT)", data=report, file_name="relatorio_rag.txt", mime="text/plain")
                # if scatter/images are available, offer a ZIP with report + plots
                imgs = st.session_state.get('last_rag_images')
                if imgs:
                    try:
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, mode='w') as zf:
                            # write report
                            zf.writestr('relatorio_rag.txt', report)
                            # write images/html
                            for name, data in imgs.items():
                                # data may be bytes already
                                if isinstance(data, str):
                                    data_b = data.encode('utf-8')
                                else:
                                    data_b = data
                                zf.writestr(name, data_b)
                        buf.seek(0)
                        zip_bytes = buf.read()
                        st.download_button("Baixar Relatório + Gráficos (ZIP)", data=zip_bytes, file_name="relatorio_rag_with_plots.zip", mime="application/zip")
                    except Exception as e:
                        st.warning(f"Não foi possível criar arquivo ZIP das imagens: {e}")
        else:
            st.info("KB não construída. Faça upload do CSV e construa a KB para gerar relatórios.")

    # Tab 2: KB / Debug
    with tabs[2]:
        st.header("KB / Debug")
        st.markdown("Informações e inspeção da KB (se construída).")
        if st.session_state.kb:
            st.write(f"Número de documentos indexados: {len(st.session_state.kb['docs'])}")
            st.write("Exemplo de documento (1):")
            st.write(st.session_state.kb["docs"][0])
            st.write("Metadatas (exemplo):")
            st.write(st.session_state.kb["metadatas"][0])
            st.markdown("---")
            st.subheader("Perguntas livres à KB (RAG)")
            q_test = st.text_area("Digite uma pergunta livre para a KB", value="Quais sinais indicam risco alto de conflito?", key='q_test', height=120)
            cols = st.columns(3)
            if cols[0].button("Recuperar (só evidências)"):
                retrieved = retrieve_top_k(st.session_state.kb, q_test, top_k=5)
                st.write("Top retrieved:")
                for r in retrieved:
                    st.markdown(f"- score={r['score']:.3f} — {r['doc'][:400]}...")
            if cols[1].button("Perguntar à KB (RAG)"):
                with st.spinner("Recuperando evidências e gerando resposta..."):
                    result = answer_query_using_kb(st.session_state.kb, q_test, top_k=5)
                st.subheader("Resposta (RAG)")
                st.write(result.get("answer", "(sem resposta)"))
                st.markdown(f"Modo: {result.get('mode')}")
                if result.get('prediction'):
                    pred = result['prediction']
                    st.subheader("Predição Heurística (RAG)")
                    st.write(f"Desfecho provável: {pred['top']} — probabilidades: {pred['probs']}")
                    st.write(pred['explanation'])
                if result.get('model_prediction'):
                    mp = result['model_prediction']
                    st.subheader("Predição do Modelo Treinado")
                    st.write(f"Probabilidade de escalada (6m): {mp.get('probability'):.3f} — Predição: {mp.get('prediction')}")
                st.subheader("Evidências recuperadas")
                for r in result.get("retrieved", []):
                    st.markdown(f"- score={r['score']:.3f} — {r['doc'][:400]}...")
            if cols[2].button("Limpar"):
                # small UX: clear the text area by rerunning with empty value (Streamlit doesn't allow direct clearing, so use session_state)
                try:
                    st.session_state['q_test'] = ''
                except Exception:
                    pass
        else:
            st.info("KB ainda não construída — faça upload do CSV e clique em 'Construir KB' na barra lateral.")

    st.markdown("---")
    st.caption("MVP RAG: a base CSV é usada como 'memória' / rede de conhecimento. Para produção: integrar LLaMA-3/endpoint próprio e persistir embeddings/index.")

else:
    # Streamlit não está disponível in this environment; UI will be disabled when importing the module for backend tests
    pass
