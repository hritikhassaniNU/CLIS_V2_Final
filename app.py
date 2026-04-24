"""
CLIS V2 — Clinical Literature Intelligence System
===================================================
Student: Hritik Hassani | Northeastern University
Course:  INFO 7375 — Generative AI & Prompt Engineering

Production UI — clean medical-grade design
4 tabs: Clinical Search · ICD-10 Coding · Evaluation · Analytics
"""

import streamlit as st
import numpy as np
import torch, torch.nn as nn
import pickle, os, sys, time, json, uuid, re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CLIS V2 | Clinical Intelligence",
    page_icon="⚕", layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DB_PATH      = os.path.join(ROOT, "clis_data.db")
BANDIT_DB    = os.path.join(ROOT, "clis_bandit.db")

# ── Design System ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"],
.stMarkdown, p, div, label {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stApp,
.main,
section[data-testid="stMain"],
[data-testid="stAppViewContainer"],
[data-testid="block-container"],
[data-testid="stAppViewBlockContainer"] {
    background: #f4f6f9 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stText"],
.stApp h1, .stApp h2, .stApp h3 { color: #1a2332; }

[data-testid="stSidebar"] {
    background: #0c1821 !important;
    border-right: 1px solid #1e3448;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stButton button {
    background: #111f2e !important; border: 1px solid #1e3448 !important;
    color: #7eb8a4 !important; border-radius: 6px !important;
    font-size: 0.75rem !important; padding: 0.4rem 0.7rem !important;
    width: 100% !important; text-align: left !important; transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #1a3040 !important; border-color: #2a9d7f !important; color: #9fe1cb !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 2px solid #dde3ee; background: transparent; margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 0.84rem; font-weight: 500;
    color: #6b7a8d !important; padding: 0.7rem 1.5rem;
    border-bottom: 2px solid transparent; margin-bottom: -2px; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0a5c45 !important; border-bottom: 2px solid #0a5c45 !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #0a5c45 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #dde3ee !important;
}

.stRadio > div { background: transparent !important; }
.stRadio label, .stRadio span, .stRadio p,
[role="radiogroup"] span, [role="radiogroup"] label,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #1a2332 !important; font-size: 0.84rem !important;
}

.stTextArea textarea {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important; color: #1a2332 !important;
    background: #ffffff !important; border: 1.5px solid #ccd5e0 !important;
    border-radius: 10px !important; padding: 0.75rem 1rem !important;
    line-height: 1.6 !important; caret-color: #0a5c45 !important;
}
.stTextArea textarea:focus {
    border-color: #0a5c45 !important; box-shadow: 0 0 0 3px rgba(10,92,69,0.1) !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: #9aa5b4 !important; font-style: italic; }
.stTextInput input {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important; color: #1a2332 !important;
    background: #ffffff !important; border: 1.5px solid #ccd5e0 !important;
    border-radius: 8px !important; padding: 0.55rem 0.9rem !important; caret-color: #0a5c45 !important;
}
.stTextInput input:focus {
    border-color: #0a5c45 !important; box-shadow: 0 0 0 3px rgba(10,92,69,0.1) !important; outline: none !important;
}
.stTextInput input::placeholder { color: #9aa5b4 !important; }
.stTextArea label, .stTextInput label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    color: #3d4d5c !important; font-weight: 500 !important; font-size: 0.82rem !important;
}

.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.83rem !important;
    border-radius: 8px !important; padding: 0.5rem 1.3rem !important;
    transition: all 0.18s ease !important;
    background: #ffffff !important;
    color: #0a5c45 !important;
    border: 1.5px solid #0a5c45 !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #0a5c45 !important; color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(10,92,69,0.2) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled { background: #f0f2f5 !important; color: #9aa5b4 !important; border-color: #dde3ee !important; transform: none !important; }
/* Primary action buttons (Search, Look up, Run) */
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="column"]:last-child .stButton > button {
    background: #0a5c45 !important; color: #ffffff !important; border: none !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #dde3ee; border-radius: 10px; padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important; color: #6b7a8d !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    font-size: 1.55rem !important; font-weight: 700 !important; color: #0c1821 !important;
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dde3ee !important;
    border-radius: 10px !important;
    margin-bottom: 0.6rem;
    overflow: hidden;
}

[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #1a2332 !important;
    background: #f8fafc !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    cursor: pointer !important;
}
[data-testid="stExpander"] summary p,
.streamlit-expanderHeader p {
    font-size: 0.84rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1a2332 !important;
}
[data-testid="stExpander"] summary:hover,
.streamlit-expanderHeader:hover { background: #f0f4f8 !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 1rem 1.2rem 1.6rem;
    background: #ffffff;
    border-top: 1px solid #f0f2f5;
}

.stProgress > div > div { background-color: #0a5c45 !important; }

.stSuccess > div { background: #f0faf5 !important; border-left: 3px solid #0a5c45 !important; color: #1a3028 !important; }
.stWarning > div { background: #fef9f0 !important; border-left: 3px solid #d4820a !important; color: #3d2800 !important; }
.stError   > div { background: #fef0f0 !important; border-left: 3px solid #c0392b !important; color: #3d0a0a !important; }
.stInfo    > div { background: #f0f5ff !important; border-left: 3px solid #2d6fb3 !important; color: #0a1a3d !important; }

hr { border-color: #dde3ee !important; margin: 1.2rem 0 !important; }
#MainMenu{visibility:hidden} footer{visibility:hidden} .stDeployButton{display:none}

.clis-header {
    background: linear-gradient(135deg, #0c1821 0%, #0a3d2e 55%, #0c5a42 100%) !important;
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 1.5rem; overflow: hidden;
}
.clis-header * { color: inherit !important; }
.header-title { font-size: 1.9rem !important; font-weight: 700 !important; color: #f0faf5 !important; letter-spacing: -0.03em; margin: 0; line-height: 1.2; }
.header-sub   { font-size: 0.85rem !important; color: #7eb8a4 !important; margin: 0.35rem 0 0; font-weight: 300 !important; }
.header-badges { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 6px; }
.hbadge {
    background: rgba(29,158,117,0.2) !important; border: 1px solid rgba(29,158,117,0.5) !important;
    color: #c0eed8 !important; font-size: 0.7rem !important; font-weight: 500 !important;
    padding: 4px 12px !important; border-radius: 20px !important;
    font-family: 'IBM Plex Mono', monospace !important; letter-spacing: 0.02em !important;
    display: inline-block !important;
}

.section-label {
    font-size: 0.69rem; font-weight: 700; color: #6b7a8d;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.4rem 0 0.7rem; border-bottom: 1px solid #edf0f5; padding-bottom: 0.4rem;
}

.pipeline-step {
    display: flex; align-items: flex-start; gap: 12px;
    background: #ffffff; border: 1px solid #dde3ee;
    border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;
}
.step-circle {
    width: 28px; height: 28px; border-radius: 50%;
    background: #0a5c45; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.74rem; font-weight: 700; flex-shrink: 0; margin-top: 1px;
}
.step-name   { font-size: 0.85rem; font-weight: 600; color: #0c1821; }
.step-detail { font-size: 0.74rem; color: #6b7a8d; margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }

.summary-box {
    background: #f0faf5; border: 1.5px solid #b8e8d4;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin: 1rem 0;
}
.summary-label { font-size: 0.69rem; font-weight: 700; color: #0a5c45; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
.summary-text  { font-size: 0.88rem; color: #1a3028 !important; line-height: 1.78; }

.ground-ok   { background: #f0faf5; border-left: 3px solid #0a5c45; padding: 8px 12px; margin: 5px 0; border-radius: 0 8px 8px 0; }
.ground-fail { background: #fef0f0; border-left: 3px solid #c0392b; padding: 8px 12px; margin: 5px 0; border-radius: 0 8px 8px 0; }

.code-pill {
    display: inline-block; background: #e8f8f1; border: 1px solid #b8e8d4;
    color: #0a5c45 !important; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem; font-weight: 600; padding: 5px 14px; border-radius: 7px; margin: 3px 5px 3px 0;
}

.section-card {
    background: #f8fafc; border: 1px solid #dde3ee;
    border-radius: 8px; padding: 9px 13px; margin-bottom: 6px;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.section-id {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
    color: #0a5c45 !important; background: #e8f8f1; padding: 2px 8px; border-radius: 4px; flex-shrink: 0;
}

.grade-pill { display: inline-block; font-size: 0.7rem; font-weight: 700; padding: 2px 10px; border-radius: 20px; font-family: 'IBM Plex Mono', monospace; }
.grade-A { background: #e8f8f1; color: #0a5c45 !important; }
.grade-B { background: #fef6e4; color: #92600a !important; }
.grade-C { background: #fdecea; color: #9c2b2b !important; }
.grade-D { background: #f0f0f0; color: #5a5a5a !important; }

.pico-box { background: #f0f6ff; border: 1px solid #c8d8f0; border-radius: 8px; padding: 0.75rem 1rem; margin: 8px 0; }
.pico-label { font-size: 0.68rem; font-weight: 700; color: #2d6fb3 !important; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 3px; font-family: 'IBM Plex Mono', monospace; }
.pico-text  { font-size: 0.9rem; color: #1a2a40 !important; font-weight: 500; }

.empty-state { text-align: center; padding: 4rem 2rem; background: #ffffff; border: 1.5px dashed #ccd5e0; border-radius: 14px; margin: 0.5rem 0; }
.empty-icon  { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #0c1821 !important; margin-bottom: 0.35rem; }
.empty-sub   { font-size: 0.82rem; color: #6b7a8d !important; }

.disclaimer { background: #fffbf0; border: 1px solid #f0d080; border-radius: 8px; padding: 8px 14px; margin-top: 1rem; font-size: 0.75rem; color: #7a5c00 !important; }

.rlhf-bar  { height: 6px; background: #e5e9f0; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.rlhf-fill { height: 100%; background: linear-gradient(90deg, #0a5c45, #2dce89); border-radius: 3px; }

.bar-wrap { background: #e8edf5; border-radius: 4px; height: 5px; margin-top: 6px; }
.bar-fill { height: 5px; border-radius: 4px; background: #0a5c45; }

.sidebar-brand { padding: 1.2rem 1rem 0.8rem; border-bottom: 1px solid #1e3448; margin-bottom: 1rem; }
.sidebar-section { font-size: 0.64rem; font-weight: 700; color: #4a6070 !important; text-transform: uppercase; letter-spacing: 0.1em; margin: 1rem 0 0.5rem; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }
.dot-green  { background: #2dce89; box-shadow: 0 0 5px rgba(45,206,137,0.5); }
.dot-yellow { background: #f6a623; }
.dot-red    { background: #e74c3c; }
.status-row { display: flex; align-items: center; padding: 3px 0; font-size: 0.76rem; }

.tc-pass { background: #f0faf5; border: 1px solid #b8e8d4; border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.3rem; }
.tc-fail { background: #fef0f0; border: 1px solid #f5c6c6; border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.3rem; }

.stat-card { background: #ffffff; border: 1px solid #dde3ee; border-radius: 10px; padding: 1rem 1.2rem; }
.stat-value { font-size: 1.8rem; font-weight: 700; color: #0c1821; line-height: 1; }
.stat-label { font-size: 0.69rem; color: #6b7a8d; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }
</style>

""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CONTEXTS  = {0:"Drug efficacy", 1:"Epidemiology", 2:"Mechanism of action", 3:"Treatment comparison"}
ARM_NAMES = {0:"MeSH + RCT filter", 1:"Keyword + Date range",
             2:"Author + Journal filter", 3:"Boolean AND + Population", 4:"Systematic review filter"}

SAMPLE_Qs = [
    "Is metformin effective for glycemic control in type 2 diabetes?",
    "SGLT2 inhibitors in heart failure with preserved ejection fraction",
    "Compare ACE inhibitors vs ARBs for heart failure outcomes",
    "GLP-1 agonists for obesity — current evidence",
    "Aspirin primary prevention in adults over 60",
]
ICD_SAMPLES = [
    "Type 2 diabetes with CKD stage 3 and hypertension",
    "Hypertensive heart disease with chronic systolic heart failure",
    "Septic shock due to MRSA with acute kidney injury",
    "COPD with acute exacerbation and pneumonia",
    "Morbid obesity BMI 42 on long-term insulin",
]

# ── Policy network ────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self,sd,na,hd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(sd,hd),nn.ReLU(),nn.Linear(hd,hd),nn.ReLU(),nn.Linear(hd,na))
    def forward(self,x): return torch.softmax(self.net(x),dim=-1)
    def probs(self,s): return self.forward(torch.FloatTensor(s).unsqueeze(0)).detach().numpy()[0]

def gv(a,k,d=None): return a.get(k,d) if isinstance(a,dict) else getattr(a,k,d)

# ── Loaders ───────────────────────────────────────────────────
@st.cache_resource
def load_rl():
    m={"ok":False}
    try:
        bp=os.path.join(ROOT,"models","bandit_policy.pkl")
        rp=os.path.join(ROOT,"models","reinforce_policy.pkl")
        if os.path.exists(bp):
            with open(bp,"rb") as f: bd=pickle.load(f)
            m["bm"]=np.array(bd["ucb_mean_rewards"]); m["bperf"]=bd.get("performance",{})
        if os.path.exists(rp):
            ck=torch.load(rp,map_location="cpu"); cfg=ck["config"]
            net=PolicyNet(cfg["state_dim"],cfg["n_actions"],cfg["hidden_dim"])
            net.load_state_dict(ck["model_state_dict"]); net.eval()
            m["pnet"]=net; m["rperf"]=ck.get("performance",{})
        m["ok"]=True
    except Exception as e: m["err"]=str(e)
    return m

@st.cache_resource
def load_icd():
    try:
        from icd10_rag_engine import ICD10RAGEngine
        return ICD10RAGEngine(groq_api_key=GROQ_API_KEY, db_path=DB_PATH)
    except: return None

@st.cache_resource
def load_bandit():
    try:
        from persistent_bandit import PersistentBandit
        return PersistentBandit(db_path=BANDIT_DB)
    except: return None

@st.cache_resource
def load_grounder():
    try:
        from citation_grounder import CitationGrounder
        return CitationGrounder()
    except: return None

@st.cache_resource
def load_evaluator():
    try:
        from benchmark_evaluator import BenchmarkEvaluator
        return BenchmarkEvaluator(groq_api_key=GROQ_API_KEY)
    except: return None

# ── Helpers ───────────────────────────────────────────────────
def classify_ctx(q):
    q=q.lower()
    if any(w in q for w in ["effective","efficacy","drug","dose","therapy","treatment"]): return 0
    if any(w in q for w in ["prevalence","incidence","rate","risk","epidem"]): return 1
    if any(w in q for w in ["mechanism","pathway","inhibit","receptor","molecular"]): return 2
    return 3

def rank_articles(pnet,articles):
    grades=[gv(a,"grade_val",0.5) for a in articles]
    recs=[gv(a,"recency_score",0.5) for a in articles]
    rels=[gv(a,"relevance_score",0.5) for a in articles]
    while len(grades)<5: grades.append(0.3); recs.append(0.5); rels.append(0.5)
    state=np.array([np.mean(grades),np.max(grades),np.var(grades),np.mean(recs),np.mean(rels),sum(1 for g in grades if g>=0.9)/5],dtype=np.float32)
    probs=pnet.probs(state); n=len(articles)
    order=np.argsort(probs[:n])[::-1]
    return [{"rank":ri+1,"title":gv(articles[idx],"title",""),"abstract":gv(articles[idx],"abstract",""),
             "grade":gv(articles[idx],"grade","C"),"grade_val":gv(articles[idx],"grade_val",0.5),
             "year":str(gv(articles[idx],"pub_year",gv(articles[idx],"year",""))),"pmid":str(gv(articles[idx],"pmid","")),
             "url":gv(articles[idx],"url",""),"recency_score":gv(articles[idx],"recency_score",0.5),
             "relevance_score":gv(articles[idx],"relevance_score",0.5),"policy_prob":float(probs[idx])}
            for ri,idx in enumerate(order)]

def grade_articles(articles,question):
    try:
        from grade_evidence_tool import GradeEvidenceTool
        grader=GradeEvidenceTool(groq_api_key=GROQ_API_KEY)
        for art in articles:
            ab=gv(art,"abstract",gv(art,"title",""))
            if ab:
                a=grader.grade(ab,question,gv(art,"title",""))
                if isinstance(art,dict): art["grade_val"]=a.grade_score; art["grade"]=a.grade_level[0]
                else: art.grade_val=a.grade_score; art.grade=a.grade_level[0]
        return articles,f"GRADE ({'Groq LLM' if grader.client else 'rule-based'})"
    except: return articles,"Rule-based"

def llm_summary(question,ranked):
    top3=ranked[:3]
    arts="\n".join([f"[{a['grade']}] {a['title']} ({a.get('year','')}) PMID:{a.get('pmid','')}" for a in top3])
    if GROQ_API_KEY and GROQ_API_KEY!="your_groq_api_key_here":
        try:
            from groq import Groq
            prompt=f"""You are a clinical evidence synthesizer for physicians.
Clinical question: {question}
Top-ranked evidence:
{arts}
Write exactly 3 sentences:
1. Main clinical finding from highest-grade evidence (cite PMID)
2. Evidence strength, limitations, and key caveats
3. Actionable clinical implication
Under 130 words. Clinically precise."""
            resp=Groq(api_key=GROQ_API_KEY).chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"user","content":prompt}],temperature=0.15,max_tokens=220)
            return resp.choices[0].message.content.strip(),True
        except: pass
    top=ranked[0] if ranked else {}
    gmap={"A":"high-quality RCT/meta-analysis","B":"moderate-quality cohort","C":"low-quality observational","D":"expert opinion"}
    return (f"Based on {len(ranked)} retrieved articles, the strongest evidence for '{question[:60]}' is {gmap.get(top.get('grade','C'),'moderate-quality')}. "
            f"The top-ranked study — '{top.get('title','N/A')[:80]}' ({top.get('year','')}) — was prioritised by the REINFORCE agent. "
            f"Physician review is required before applying these findings to individual patient decisions."),False

def detect_contradictions(question,ranked):
    if not GROQ_API_KEY or GROQ_API_KEY=="your_groq_api_key_here": return False,"Groq key required.",[]
    arts=[a for a in ranked[:3] if len(str(a.get("abstract","")))>30]
    if len(arts)<2: return False,"Need ≥2 abstracts to check.",[]
    arts_text="\n\n".join([f"Article {i+1} [Grade {a['grade']}]: {a['title']} — {str(a.get('abstract',''))[:280]}" for i,a in enumerate(arts)])
    prompt=f"Clinical question: {question}\nArticles:\n{arts_text}\nDo any articles contradict each other? Reply ONLY valid JSON:\n{{\"contradicts\":true/false,\"severity\":\"none\"/\"minor\"/\"major\",\"explanation\":\"1 sentence\",\"pairs\":[]}}"
    try:
        from groq import Groq
        raw=Groq(api_key=GROQ_API_KEY).chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"user","content":prompt}],temperature=0.05,max_tokens=180).choices[0].message.content.strip()
        raw=re.sub(r"```json|```","",raw).strip()
        d=json.loads(raw)
        return bool(d.get("contradicts",False)),str(d.get("explanation","")),d.get("pairs",[])
    except Exception as e: return False,f"Check unavailable: {str(e)[:50]}",[]

def export_report(question,ranked,summary,citations=None):
    lines=["CLIS V2 — CLINICAL EVIDENCE REPORT","="*60,
           f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}","",
           "CLINICAL QUESTION","-"*40,question,"",
           "EVIDENCE SUMMARY","-"*40,summary,"","RANKED ARTICLES","-"*40]
    for a in ranked:
        gfull={"A":"High","B":"Moderate","C":"Low","D":"Very Low"}.get(a.get("grade","C"),"Low")
        lines+=[f"#{a['rank']} [{gfull}] {a['title']}",
                f"   PMID:{a.get('pmid','')}  Year:{a.get('year','')}  Grade:{a.get('grade_val',0):.2f}  RL prob:{a.get('policy_prob',0):.3f}",
                f"   {a.get('url','N/A')}",""]
    if citations:
        lines+=["","CITATION GROUNDING","-"*40,f"Faithfulness: {citations.faithfulness_score:.2f}",""]
        for s in citations.sentences:
            lines+=[f"[{'GROUNDED' if s.is_grounded else 'UNGROUNDED'}] {s.text[:100]}",
                    f"   Source: {s.source_title[:60]} (PMID {s.pmid})",""]
    lines+=["="*60,"DISCLAIMER: Decision support only. Verify all citations. Apply clinical judgment.","="*60]
    return "\n".join(lines)

def grade_css(g):
    return {"A":"grade-A","H":"grade-A","B":"grade-B","M":"grade-B","C":"grade-C","L":"grade-C","D":"grade-D","V":"grade-D"}.get(g,"grade-C")

def grade_label(g):
    return {"A":"High","H":"High","B":"Moderate","M":"Moderate","C":"Low","L":"Low","D":"Very Low","V":"Very Low"}.get(g,g)

# ── Session state ─────────────────────────────────────────────
for k,v in [("history",[]),("eval_results",None),("session_id",str(uuid.uuid4())[:8]),("cur_q","")]:
    if k not in st.session_state: st.session_state[k]=v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand">
    <div style="display:flex;align-items:center;gap:10px">
        <div style="width:36px;height:36px;background:#0a5c45;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem">⚕</div>
        <div>
            <div style="font-size:.95rem;font-weight:600;color:#e8f5f0;letter-spacing:-.01em">CLIS V2</div>
            <div style="font-size:.65rem;color:#4a7080;margin-top:1px">Clinical Intelligence · NEU</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)
    m=load_rl(); icd=load_icd(); bandit_obj=load_bandit()
    has_groq = GROQ_API_KEY and GROQ_API_KEY!="your_groq_api_key_here"

    statuses=[
        ("RL models","dot-green" if m["ok"] else "dot-yellow", "Bandit + REINFORCE loaded" if m["ok"] else "Demo mode"),
        ("ICD-10 RAG","dot-green" if icd else "dot-yellow", f"{len(icd.kb)} sections" if icd else "Unavailable"),
        ("Bandit DB","dot-green" if bandit_obj else "dot-yellow", f"{bandit_obj.get_stats()['total_queries']} queries" if bandit_obj else "Unavailable"),
        ("Groq LLM","dot-green" if has_groq else "dot-red","Llama 3.3 70B" if has_groq else "Add key to .env"),
    ]
    for name,dot,detail in statuses:
        st.markdown(f'<div class="status-row"><span class="status-dot {dot}"></span><span style="color:#8aabb8;font-size:.76rem">{name}</span><span style="color:#4a6070;font-size:.68rem;margin-left:auto;font-family:IBM Plex Mono">{detail}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Sample Questions</div>', unsafe_allow_html=True)
    selected_q=None
    for i,q in enumerate(SAMPLE_Qs):
        if st.button(q[:48]+"...", key=f"sidebar_sq_{i}", use_container_width=True): selected_q=q

    if st.session_state.history:
        st.markdown('<div class="sidebar-section">Recent Searches</div>', unsafe_allow_html=True)
        for h in list(reversed(st.session_state.history[-4:])):
            q_s=h["q"][:38]+"..." if len(h["q"])>38 else h["q"]
            gc={"A":"#0a5c45","B":"#92600a","C":"#9c2b2b","D":"#5a5a5a"}.get(h.get("grade","C"),"#6b7a8d")
            st.markdown(f'<div style="background:#111f2e;border:0.5px solid #1e3448;border-radius:6px;padding:6px 10px;margin-bottom:4px"><div style="font-size:.68rem;color:#4a6070;font-family:IBM Plex Mono">{h["ts"]} · <span style="color:{gc}">Grade {h.get("grade","?")}</span></div><div style="font-size:.74rem;color:#8aabb8;margin-top:2px;line-height:1.4">{q_s}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #1e3448;padding-top:.8rem;margin-top:1rem"><div style="font-size:.65rem;color:#4a6070;line-height:1.6">Hritik Hassani · NEU<br>INFO 7375 · Spring 2026<br>Session: {}</div></div>'.format(st.session_state.session_id), unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="clis-header">
    <div class="header-title">⚕ CLIS V2</div>
    <div class="header-sub">Clinical Literature Intelligence System — Production Build</div>
    <div class="header-badges">
        <span class="hbadge">UCB Bandit</span>
        <span class="hbadge">REINFORCE</span>
        <span class="hbadge">PubMed RAG</span>
        <span class="hbadge">GRADE Evidence</span>
        <span class="hbadge">Citation Grounding</span>
        <span class="hbadge">ICD-10 RAG</span>
        <span class="hbadge">RLHF Feedback</span>
        <span class="hbadge">Persistent Bandit</span>
        <span class="hbadge">Groq · Llama 3.3 70B</span>
        <span class="hbadge">Treatment Comparison</span>
        <span class="hbadge">NNT Extractor</span>
        <span class="hbadge">Knowledge Graph</span>
    </div>
</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Clinical Search","ICD-10 Coding","Advanced Analysis","Evaluation Dashboard","System Analytics"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL SEARCH
# ══════════════════════════════════════════════════════════════
with tab1:
    mode=st.radio("Input mode",["Free text","PICO builder"],horizontal=True,label_visibility="collapsed")
    question,run="",False

    if mode=="Free text":
        c1,c2=st.columns([5,1])
        with c1:
            question=st.text_area("Q",height=80,label_visibility="collapsed",
                value=selected_q or st.session_state.cur_q,
                placeholder="e.g. Is metformin effective for glycemic control in type 2 diabetes?")
        with c2:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            run=st.button("Search",use_container_width=True,key="run1")
    else:
        st.markdown('<div style="background:#f0f6ff;border:1px solid #c8d8f0;border-radius:8px;padding:.6rem 1rem;margin-bottom:.8rem;font-size:.78rem;color:#2d6fb3;font-family:IBM Plex Mono">PICO = Population · Intervention · Comparison · Outcome — the international standard for clinical questions</div>', unsafe_allow_html=True)
        r1,r2=st.columns(2)
        with r1:
            pp=st.text_input("P — Population",placeholder="e.g. adults with type 2 diabetes over 50")
            pi=st.text_input("I — Intervention",placeholder="e.g. metformin 1000mg twice daily")
        with r2:
            pc=st.text_input("C — Comparison (optional)",placeholder="e.g. placebo")
            po=st.text_input("O — Outcome",placeholder="e.g. HbA1c reduction at 12 months")
        pq=""
        if pp and pi and po:
            pq=f"In {pp}, is {pi} more effective than {pc} for {po}?" if pc else f"In {pp}, is {pi} effective for {po}?"
        if pq:
            st.markdown(f'<div class="pico-box"><div class="pico-label">Built clinical question</div><div class="pico-text">{pq}</div></div>', unsafe_allow_html=True)
        _,rc=st.columns([5,1])
        with rc: run=st.button("Search",disabled=not pq,use_container_width=True,key="run_pico")
        question=pq

    if question: st.session_state.cur_q=question

    if run and question.strip():
        models=load_rl(); bandit_obj=load_bandit(); grounder=load_grounder()
        bm=models.get("bm") if models["ok"] else None; pnet=models.get("pnet")

        st.markdown('<div class="section-label">Pipeline Execution</div>', unsafe_allow_html=True)

        # Step 1
        ctx_id=classify_ctx(question); ctx_name=CONTEXTS[ctx_id]
        st.markdown(f'<div class="pipeline-step"><div class="step-circle">1</div><div class="step-body"><div class="step-name">Context classified</div><div class="step-detail">{ctx_name} (context {ctx_id})</div></div></div>', unsafe_allow_html=True)

        # Step 2
        if bandit_obj: arm_id,ucb_score=bandit_obj.select_arm(ctx_id); arm_src="Persistent UCB bandit"
        elif bm is not None: arm_id=int(np.argmax(bm[ctx_id])); ucb_score=float(bm[ctx_id][arm_id]); arm_src="Trained bandit model"
        else: arm_id,ucb_score,arm_src=0,0.5,"Default"
        st.markdown(f'<div class="pipeline-step"><div class="step-circle">2</div><div class="step-body"><div class="step-name">UCB bandit selected query strategy</div><div class="step-detail">Arm {arm_id}: {ARM_NAMES[arm_id]} · UCB score {ucb_score:.3f} · {arm_src}</div></div></div>', unsafe_allow_html=True)

        # Step 3 — Real PubMed only
        articles=[]; pubmed_err=""
        try:
            from pubmed_retriever import PubMedRetriever
            ret=PubMedRetriever()
            with st.spinner("Querying PubMed NCBI API..."):
                qry=ret.build_query(question,arm_id); articles=ret.search(qry,max_results=6)
        except Exception as e: pubmed_err=str(e)

        if not articles:
            st.error(f"**PubMed returned no results.** {'Error: '+pubmed_err[:80] if pubmed_err else 'Try rephrasing your question with more specific clinical terms.'}\n\nCLIS V2 only shows real peer-reviewed evidence — no simulated data.")
            st.stop()

        st.markdown(f'<div class="pipeline-step"><div class="step-circle">3</div><div class="step-body"><div class="step-name">PubMed articles retrieved</div><div class="step-detail">{len(articles)} real articles · "{qry[:55]}" · NCBI E-utilities API</div></div></div>', unsafe_allow_html=True)

        # Step 4
        articles,grade_method=grade_articles(articles,question)
        avg_grade=np.mean([gv(a,"grade_val",0.5) for a in articles])
        st.markdown(f'<div class="pipeline-step"><div class="step-circle">4</div><div class="step-body"><div class="step-name">Evidence graded</div><div class="step-detail">{grade_method} · Average grade score {avg_grade:.3f}</div></div></div>', unsafe_allow_html=True)

        # Step 5
        if pnet: ranked=rank_articles(pnet,articles); rl_src="Trained REINFORCE policy"
        else:
            sa=sorted(articles,key=lambda x:gv(x,"grade_val",0.5),reverse=True)
            ranked=[{"rank":i+1,"title":gv(a,"title",""),"abstract":gv(a,"abstract",""),"grade":gv(a,"grade","C"),"grade_val":gv(a,"grade_val",0.5),"year":str(gv(a,"pub_year",gv(a,"year",""))),"pmid":str(gv(a,"pmid","")),"url":gv(a,"url",""),"recency_score":gv(a,"recency_score",0.5),"relevance_score":gv(a,"relevance_score",0.5),"policy_prob":round(0.35-i*0.05,3)} for i,a in enumerate(sa)]
            rl_src="Grade-sorted fallback"
        st.markdown(f'<div class="pipeline-step"><div class="step-circle">5</div><div class="step-body"><div class="step-name">REINFORCE ranked evidence</div><div class="step-detail">{rl_src}</div></div></div>', unsafe_allow_html=True)

        # Step 6
        with st.spinner("Generating clinical summary (Groq Llama 3.3 70B)..."):
            summary,used_llm=llm_summary(question,ranked)
        citations=grounder.ground(summary,articles,question) if grounder and used_llm else None
        faith=citations.faithfulness_score if citations else 0.5
        st.markdown(f'<div class="pipeline-step"><div class="step-circle">6</div><div class="step-body"><div class="step-name">Summary generated + citations grounded</div><div class="step-detail">{"Groq Llama 3.3 70B" if used_llm else "Template fallback"} · Faithfulness {faith:.2f} ({citations.grounded_count if citations else "?"}/{(citations.grounded_count+citations.ungrounded_count) if citations else "?"} sentences grounded)</div></div></div>', unsafe_allow_html=True)

        # Persist
        top_grade=ranked[0]["grade"] if ranked else "C"
        reward=gv(ranked[0],"grade_val",0.5)*0.7+avg_grade*0.3
        if bandit_obj: bandit_obj.update(ctx_id,arm_id,reward,session_id=st.session_state.session_id,query=question[:200],top_grade=top_grade)

        # ── RESULTS ───────────────────────────────────────────
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Articles retrieved",len(articles))
        c2.metric("Top grade",grade_label(top_grade))
        c3.metric("Avg grade score",f"{avg_grade:.3f}")
        c4.metric("Query arm",f"Arm {arm_id}")
        c5.metric("Faithfulness",f"{faith:.2f}")

        # RLHF
        st.markdown("""<div style="background:#f8fafc;border:1px solid #e5e9f0;border-radius:10px;padding:0.6rem 1rem;margin:0.8rem 0;display:flex;align-items:center;gap:12px">
    <span style="font-size:0.78rem;font-weight:600;color:#3d4d5c">Was this result helpful?</span>
    <span style="font-size:0.68rem;color:#9aa5b4;font-family:'IBM Plex Mono',monospace">RLHF feedback updates bandit rewards</span>
</div>""", unsafe_allow_html=True)
        fb1,fb2,fb_spacer=st.columns([1,1,5])
        with fb1:
            if st.button("👍 Helpful", key="thu"):
                if bandit_obj: bandit_obj.apply_rlhf(ctx_id,arm_id,+1)
                st.success("Reward +0.15 applied")
        with fb2:
            if st.button("👎 Not helpful", key="thd"):
                if bandit_obj: bandit_obj.apply_rlhf(ctx_id,arm_id,-1)
                st.warning("Reward -0.15 applied")

        # Summary
        llm_src="Groq · Llama 3.3 70B" if used_llm else "Template fallback"
        st.markdown(f"""
<div class="summary-box">
    <div class="summary-label">Clinical Evidence Summary &nbsp;·&nbsp; {llm_src}</div>
    <div class="summary-text">{summary}</div>
</div>""", unsafe_allow_html=True)

        # Citation grounding
        if citations:
            faith_pct=int(citations.faithfulness_score*100)
            with st.expander(f"Citation grounding - {citations.grounded_count}/{citations.grounded_count+citations.ungrounded_count} sentences verified (faithfulness {citations.faithfulness_score:.2f})", expanded=citations.ungrounded_count>0):
                if citations.warning:
                    st.warning(citations.warning)
                for s in citations.sentences:
                    if s.is_grounded:
                        pmid_link=f'<a href="{s.url}" target="_blank" style="color:#0a5c45;font-weight:500">{s.pmid}</a>' if s.url else s.pmid
                        detail=f"PMID {pmid_link} · {s.source_title[:55]} · overlap {s.overlap_score:.2f}"
                        st.markdown(f'<div class="ground-ok"><div style="font-size:.82rem;color:#0c3020;font-weight:500">Grounded: {s.text}</div><div style="font-size:.7rem;color:#5a8070;margin-top:2px;font-family:IBM Plex Mono">{detail}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ground-fail"><div style="font-size:.82rem;color:#7a1a1a;font-weight:500">Ungrounded: {s.text}</div><div style="font-size:.7rem;color:#9a5050;margin-top:2px;font-family:IBM Plex Mono">Could not match to source (overlap {s.overlap_score:.2f})</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Ranked Evidence — REINFORCE Policy Ordering</div>', unsafe_allow_html=True)
        max_prob = max((a['policy_prob'] for a in ranked), default=0.35)
        for art in ranked:
            g        = str(art.get('grade', 'C'))
            is_top   = art['rank'] == 1
            pmid     = str(art.get('pmid', ''))
            url      = art.get('url', '') or (f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else '')
            title    = str(art.get('title', ''))
            abstract = str(art.get('abstract', ''))
            yr       = str(art.get('year', ''))
            gfull2   = grade_label(g)
            pp       = min(int(art['policy_prob'] / max(max_prob, 0.01) * 100), 100)

            grade_bg   = {'A':'#e8f8f1','H':'#e8f8f1','B':'#fef6e4','M':'#fef6e4'}.get(g,'#fdecea')
            grade_clr  = {'A':'#0a5c45','H':'#0a5c45','B':'#92600a','M':'#92600a'}.get(g,'#9c2b2b')
            top_txt    = ' [TOP PICK]' if is_top else ''
            t_short    = title[:85] + ('...' if len(title) > 85 else '')
            exp_label  = f'#{art["rank"]} [{gfull2}] {t_short}{top_txt}'
            top_badge  = f'<span style="background:#e8f8f1;color:#0a5c45;border:1px solid #0a5c45;padding:2px 10px;border-radius:4px;font-size:0.68rem;font-weight:700;letter-spacing:.03em">TOP PICK</span>' if is_top else ''
            pmid_html  = f'<a href="{url}" target="_blank" style="color:#0a5c45;text-decoration:none;font-weight:500">PMID {pmid}</a>' if pmid and url else (f'PMID {pmid}' if pmid else '')
            pubmed_btn = f'<a href="{url}" target="_blank" style="display:inline-block;background:#0a5c45;color:#fff;font-size:0.72rem;font-weight:600;padding:4px 14px;border-radius:6px;text-decoration:none;letter-spacing:.02em">Open in PubMed &rarr;</a>' if url else ''

            with st.expander(exp_label, expanded=is_top):
                st.markdown(f'''<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px">
    <span style="background:{grade_bg};color:{grade_clr};padding:3px 12px;border-radius:5px;font-size:0.76rem;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:.02em">{g} &middot; {gfull2}</span>
    {top_badge}
    <span style="font-size:0.76rem;color:#6b7a8d;font-family:'IBM Plex Mono',monospace;margin-left:auto">{pmid_html}{(" &middot; " + yr) if yr else ""}</span>
</div>
<div style="font-size:0.95rem;font-weight:600;color:#0c1821;line-height:1.55;margin-bottom:10px">{title}</div>''', unsafe_allow_html=True)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("RL Probability", f"{art['policy_prob']:.3f}")
                mc2.metric("Grade Score", f"{art['grade_val']:.2f}")
                mc3.metric("Recency", f"{art.get('recency_score',0.5):.2f}")
                mc4.metric("Relevance", f"{art.get('relevance_score',0.5):.2f}")

                st.markdown(f'''<div style="margin:8px 0 14px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:0.68rem;font-weight:600;color:#6b7a8d;text-transform:uppercase;letter-spacing:.06em">Policy confidence</span>
        <span style="font-size:0.72rem;font-weight:600;color:#0a5c45;font-family:'IBM Plex Mono',monospace">{pp}%</span>
    </div>
    <div style="background:#e5e9f0;border-radius:4px;height:6px;overflow:hidden">
        <div style="height:6px;border-radius:4px;background:linear-gradient(90deg,#0a5c45,#2dce89);width:{pp}%"></div>
    </div>
</div>''', unsafe_allow_html=True)

                if abstract and len(abstract) > 40 and 'Abstract for:' not in abstract:
                    st.markdown(f'''<div style="background:#f8fafc;border:1px solid #e5e9f0;border-radius:8px;padding:0.9rem 1.1rem;margin-top:4px">
    <div style="font-size:0.7rem;font-weight:700;color:#6b7a8d;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Abstract</div>
    <div style="font-size:0.84rem;color:#2a3a4a;line-height:1.75">{abstract[:700]}{"..." if len(abstract)>700 else ""}</div>
</div>''', unsafe_allow_html=True)

                if pubmed_btn:
                    st.markdown(f'<div style="margin-top:10px">{pubmed_btn}</div>', unsafe_allow_html=True)

        # Contradiction check
        st.markdown('<div class="section-label">Contradiction Check</div>', unsafe_allow_html=True)
        with st.spinner("Analysing for conflicting evidence..."):
            contradicts,explanation,pairs=detect_contradictions(question,ranked)
        if not has_groq:
            st.info("Add GROQ_API_KEY to .env to enable contradiction detection.")
        elif contradicts:
            ph="".join([f'<div style="font-size:.78rem;color:#7a1a1a;padding:3px 0;border-bottom:1px solid #f0c8c8">{p}</div>' for p in pairs])
            st.markdown(f'<div style="background:#fef0f0;border:1.5px solid #e88080;border-radius:10px;padding:1rem 1.2rem"><div style="font-weight:600;color:#9c2b2b;font-size:.86rem;margin-bottom:6px">⚠ Contradictions detected</div><div style="font-size:.84rem;color:#5a1a1a;line-height:1.6;margin-bottom:8px">{explanation}</div>{ph}<div style="font-size:.72rem;color:#9c2b2b;margin-top:8px;font-style:italic">Clinician review required before applying contradictory evidence to patient decisions.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#f0faf5;border:1px solid #b8e8d4;border-radius:10px;padding:.9rem 1.1rem"><div style="font-weight:600;color:#0a5c45;font-size:.86rem;margin-bottom:4px">✓ No major contradictions detected</div><div style="font-size:.83rem;color:#1a4030;line-height:1.6">{explanation}</div></div>', unsafe_allow_html=True)

        # Export
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        report=export_report(question,ranked,summary,citations)
        st.download_button("⬇ Download clinical report",data=report,file_name=f"clis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",mime="text/plain")

        st.session_state.history.append({"q":question,"arm":arm_id,"grade":top_grade,"ts":datetime.now().strftime("%H:%M")})

    elif run and not question.strip():
        st.warning("Please enter a clinical question first.")
    else:
        st.markdown("""
<div class="empty-state">
    <div class="empty-icon">⚕</div>
    <div class="empty-title">Enter a clinical question to begin</div>
    <div class="empty-sub">Real PubMed evidence only · GRADE-graded · RL-ranked · Every claim citation-grounded</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — ICD-10 CODING
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">ICD-10-CM Coding Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#f0f6ff;border:1px solid #c8d8f0;border-radius:8px;padding:.8rem 1.1rem;margin-bottom:1rem;font-size:.82rem;color:#1a2a50;line-height:1.6">RAG pipeline over 47 CMS ICD-10-CM FY2024 guideline sections · Groq Llama 3.3 70B synthesizes the answer · SQLite-cached for instant repeat lookups</div>', unsafe_allow_html=True)

    ci1,ci2=st.columns([5,1])
    with ci1:
        icd_q=st.text_area("Coding question",height=80,label_visibility="collapsed",
                            placeholder="e.g. Type 2 diabetes with CKD stage 3, hypertension, and systolic heart failure")
    with ci2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run_icd=st.button("Look up",use_container_width=True,key="run_icd2")

    st.markdown('<div class="section-label">Quick Lookups</div>', unsafe_allow_html=True)
    sel_icd=None
    qc=st.columns(3)
    for i,q in enumerate(ICD_SAMPLES[:3]):
        with qc[i]:
            if st.button(q[:36]+"...",key=f"icd_quick_{i}",use_container_width=True): sel_icd=q
    qc2=st.columns(2)
    for i,q in enumerate(ICD_SAMPLES[3:]):
        with qc2[i]:
            if st.button(q[:36]+"...",key=f"icd_quick2_{i}",use_container_width=True): sel_icd=q

    final_icd_q=sel_icd or (icd_q if run_icd else None)

    if final_icd_q:
        icd_engine=load_icd()
        if not icd_engine:
            st.error("ICD-10 RAG engine not loaded. Check tools/icd10_rag_engine.py is present.")
        else:
            with st.spinner("Retrieving guideline sections and generating coding answer..."):
                result=icd_engine.answer(final_icd_q)

            st.markdown('<div class="section-label">Coding Result</div>', unsafe_allow_html=True)

            # Codes
            if result.primary_codes:
                codes_html="".join([f'<span class="code-pill">{code}</span>' for code in result.primary_codes])
                st.markdown(f'<div style="margin-bottom:1.2rem"><div style="font-size:.7rem;font-weight:600;color:#6b7a8d;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem">Relevant codes</div><div>{codes_html}</div></div>', unsafe_allow_html=True)

            # Answer
            fc_map={"High":"#0a5c45","Moderate":"#92600a","Low":"#9c2b2b"}
            fc=fc_map.get(result.confidence,"#6b7a8d")
            st.markdown(f"""
<div style="background:#f0faf5;border:1.5px solid #b8e8d4;border-radius:12px;padding:1.2rem 1.4rem">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:.8rem">
        <span style="font-size:.7rem;font-weight:600;color:#0a5c45;text-transform:uppercase;letter-spacing:.07em">Coding guidance</span>
        <span style="font-size:.68rem;color:#fff;background:#0a5c45;padding:2px 9px;border-radius:10px;font-family:IBM Plex Mono">{result.generated_by}</span>
        <span style="font-size:.68rem;color:{fc};background:#f8f9fb;border:1px solid #e5e9f0;padding:2px 9px;border-radius:10px;font-family:IBM Plex Mono">faithfulness {result.faithfulness_score:.2f}</span>
    </div>
    <div style="font-size:.86rem;color:#1a3028;line-height:1.85;white-space:pre-wrap">{result.answer}</div>
</div>""", unsafe_allow_html=True)

            # Sections
            if result.supporting_sections:
                st.markdown('<div class="section-label" style="margin-top:1.2rem">Retrieved Guideline Sections</div>', unsafe_allow_html=True)
                for sec in result.supporting_sections:
                    codes_str=", ".join(sec.get("codes",[])[:4])
                    st.markdown(f'<div class="section-card"><span class="section-id">{sec["section"]}</span><span style="font-size:.82rem;color:#1a2a40;font-weight:500">{sec["title"]}</span><span style="font-size:.7rem;color:#6b7a8d;margin-left:auto;font-family:IBM Plex Mono">{sec.get("specialty","")} · {codes_str}</span></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="disclaimer">⚠ {result.disclaimer}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="empty-state">
    <div class="empty-icon">⊕</div>
    <div class="empty-title">Enter a coding question or use a quick lookup</div>
    <div class="empty-sub">Covers 47 sections: diabetes · hypertension · CKD · sepsis · cancer · COPD · fractures · psychiatry · obstetrics</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Benchmark Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#f0f6ff;border:1px solid #c8d8f0;border-radius:8px;padding:.8rem 1.1rem;margin-bottom:1rem;font-size:.82rem;color:#1a2a50;line-height:1.6">10 automated test cases from the project proposal · TC10 is the hallucination trap — a clinical trial that does not exist. The system must refuse to fabricate results.</div>', unsafe_allow_html=True)

    cr1,cr2=st.columns([1,5])
    with cr1: run_eval=st.button("Run all 10 tests",use_container_width=True,key="run_eval2")
    with cr2: st.markdown('<div style="padding-top:10px;font-size:.8rem;color:#6b7a8d">~60 seconds with Groq · tests run against live PubMed + ICD-10 pipeline</div>', unsafe_allow_html=True)

    if run_eval:
        ev=load_evaluator()
        if not ev: st.error("Evaluator not loaded.")
        else:
            pr=None
            try:
                from pubmed_retriever import PubMedRetriever; pr=PubMedRetriever()
            except: pass
            ie=load_icd()
            pb=st.progress(0); stxt=st.empty()
            def on_prog(i,t,n):
                pb.progress((i+1)/t)
                stxt.markdown(f'<div style="font-size:.78rem;color:#6b7a8d;font-family:IBM Plex Mono">TC{i+1:02d}/{t} — {n}...</div>', unsafe_allow_html=True)
            with st.spinner("Running benchmark suite..."):
                results=ev.run_all(pr,ie,on_prog)
            pb.progress(1.0); stxt.empty()
            st.session_state.eval_results=results

    results=st.session_state.eval_results
    if results:
        from benchmark_evaluator import BenchmarkEvaluator
        ev2=load_evaluator() or BenchmarkEvaluator(); sm=ev2.summarize(results)

        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Tests passed",f"{sm['passed']}/{sm['total']}")
        c2.metric("Pass rate",f"{sm['pass_rate']}%")
        c3.metric("Avg score",f"{sm['avg_score']:.2f}")
        c4.metric("Critical tests",f"{sm['critical_passed']}/{sm['critical_total']}")
        c5.metric("Hallucination trap","PASSED ✓" if sm["hallucination_trap_passed"] else "FAILED ✗")

        # Hallucination trap highlight
        if sm["hallucination_trap_passed"]:
            st.markdown('<div style="background:#f0faf5;border:1.5px solid #b8e8d4;border-radius:10px;padding:.9rem 1.2rem;margin:.8rem 0"><div style="font-weight:600;color:#0a5c45;margin-bottom:4px">✓ Hallucination trap passed (TC10)</div><div style="font-size:.82rem;color:#1a4030">CARDIAC-PREVENT does not exist — system correctly refused to fabricate results for a non-existent trial.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#fef0f0;border:1.5px solid #e88080;border-radius:10px;padding:.9rem 1.2rem;margin:.8rem 0"><div style="font-weight:600;color:#9c2b2b;margin-bottom:4px">✗ Hallucination trap failed (TC10)</div><div style="font-size:.82rem;color:#5a1a1a">System may have fabricated results for CARDIAC-PREVENT, a trial that does not exist.</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.2rem">Individual Test Results</div>', unsafe_allow_html=True)
        for r in results:
            ok=r.passed
            with st.expander(f"TC{r.id:02d} {'PASS' if ok else 'FAIL'} - {r.name} [{r.category}]{'  CRITICAL' if r.critical else ''}", expanded=not ok):
                status_col, meta_col = st.columns([3,1])
                with status_col:
                    if ok: st.success(f"PASS — {r.name}{'  · CRITICAL' if r.critical else ''}")
                    else:  st.error(f"FAIL — {r.name}{'  · CRITICAL' if r.critical else ''}")
                with meta_col:
                    st.caption(f"Score: {r.score:.1f} · {r.latency_ms:.0f}ms")
                st.caption(f"**Query:** {r.query}")
                e_col, g_col = st.columns(2)
                with e_col:
                    st.markdown("**Expected**"); st.info(r.expected)
                with g_col:
                    st.markdown("**Got**"); st.info(r.got[:200] + ("..." if len(r.got) > 200 else ""))
                if r.notes: st.caption(f"Note: {r.notes}")

        st.markdown('<div class="section-label" style="margin-top:1.2rem">By Category</div>', unsafe_allow_html=True)
        cats=list(sm["by_category"].items()); cc=st.columns(min(len(cats),4))
        for i,(cat,cts) in enumerate(cats):
            pct=round(cts["passed"]/cts["total"]*100)
            col=cc[i%4]
            col.markdown(f'<div class="stat-card" style="margin-bottom:8px"><div style="font-size:.7rem;color:#6b7a8d;font-family:IBM Plex Mono;margin-bottom:4px">{cat}</div><div style="font-size:1.1rem;font-weight:600;color:#0c1821">{cts["passed"]}/{cts["total"]} <span style="font-size:.75rem;color:#6b7a8d;font-weight:400">({pct}%)</span></div><div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div></div>', unsafe_allow_html=True)

        st.divider()
        ed={"summary":sm,"timestamp":datetime.now().isoformat(),"results":[{"id":r.id,"name":r.name,"category":r.category,"passed":r.passed,"score":r.score,"latency_ms":r.latency_ms} for r in results]}
        st.download_button("⬇ Export benchmark results (JSON)",data=json.dumps(ed,indent=2),file_name=f"clis_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json",mime="application/json")
    else:
        st.markdown("""
<div class="empty-state">
    <div class="empty-icon">✓</div>
    <div class="empty-title">Click "Run all 10 tests" to start the evaluation</div>
    <div class="empty-sub">Tests run against the live pipeline · Results include pass/fail, latency, and category breakdown</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">System Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.82rem;color:#6b7a8d;margin-bottom:1.2rem">Live bandit learning stats, RLHF feedback, and per-context arm performance across all sessions.</div>', unsafe_allow_html=True)

    bandit_obj=load_bandit()
    if not bandit_obj:
        st.warning("Persistent bandit not loaded. Run a clinical search to initialise it.")
    else:
        stats=bandit_obj.get_stats(); lc=bandit_obj.get_learning_curve(50); state=bandit_obj.get_state()

        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Total queries",stats["total_queries"])
        c2.metric("Sessions",stats["sessions"])
        c3.metric("RLHF rated",stats["rated_queries"])
        c4.metric("Thumbs up",stats["thumbs_up"])
        c5.metric("Avg reward",f"{stats['avg_reward']:.4f}")

        st.divider()
        col_lc,col_bm=st.columns(2)

        with col_lc:
            st.markdown('<div class="section-label">Learning Curve — Last 50 Queries</div>', unsafe_allow_html=True)
            if lc:
                st.line_chart({"Smoothed reward (rolling avg)":lc},height=220)
                st.caption(f"5-point rolling average · {len(lc)} data points · updates with every query")
            else:
                st.info("Run clinical queries to populate the learning curve.")

        with col_bm:
            st.markdown('<div class="section-label">Bandit State — Mean Reward Per Strategy</div>', unsafe_allow_html=True)
            ctx_labels=["Drug efficacy","Epidemiology","Mechanism","Treatment"]
            arm_labels=["MeSH+RCT","Keyword","Author","BoolAND","SysRev"]
            for ci,ctx in enumerate(ctx_labels):
                arm_means=state[ci]; best_arm=int(np.argmax(arm_means))
                rh=f'<div style="margin-bottom:10px"><div style="font-size:.72rem;font-weight:600;color:#6b7a8d;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px">{ctx}</div><div style="display:flex;gap:4px">'
                for ai,(mean,name) in enumerate(zip(arm_means,arm_labels)):
                    ib=ai==best_arm
                    bg="#e8f8f1" if ib else "#f8f9fb"
                    br="1.5px solid #0a5c45" if ib else "1px solid #e5e9f0"
                    clr="#0a5c45" if ib else "#6b7a8d"
                    fw="600" if ib else "400"
                    rh+=f'<div style="flex:1;background:{bg};border:{br};border-radius:6px;padding:5px 4px;text-align:center"><div style="font-size:.6rem;color:{clr};font-family:IBM Plex Mono">{name}</div><div style="font-size:.82rem;color:#0c1821;font-weight:{fw};margin-top:1px">{mean:.3f}</div></div>'
                rh+="</div></div>"
                st.markdown(rh,unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-label">RLHF Feedback Impact</div>', unsafe_allow_html=True)
        if stats["rated_queries"]>0:
            tr=stats["thumbs_up"]+stats["thumbs_down"]; pp3=round(stats["thumbs_up"]/max(tr,1)*100)
            c1,c2,c3=st.columns(3)
            c1.metric("Helpful responses",stats["thumbs_up"])
            c2.metric("Not helpful",stats["thumbs_down"])
            c3.metric("Positive rate",f"{pp3}%")
            st.markdown(f'<div class="rlhf-bar"><div class="rlhf-fill" style="width:{pp3}%"></div></div>', unsafe_allow_html=True)
            st.caption("Each rating shifts the bandit reward estimate ±0.15 for the relevant context-arm pair and persists across sessions")
        else:
            st.info("No RLHF feedback yet. Use the 👍 / 👎 buttons in the Clinical Search tab after each result.")

        st.divider()
        st.markdown('<div class="section-label">Reset</div>', unsafe_allow_html=True)
        if st.button("Reset bandit to domain priors",key="reset_bandit"):
            try:
                import sqlite3 as sq3; conn=sq3.connect(BANDIT_DB)
                for tbl in ["bandit_state","feedback","sessions"]:
                    conn.execute(f"DELETE FROM {tbl}")
                conn.commit(); conn.close()
                st.cache_resource.clear()
                st.success("Bandit state reset to domain-informed priors.")
            except Exception as e: st.error(f"Reset failed: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 5 — ADVANCED ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">Advanced Analysis Tools</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#f0f6ff;border:1px solid #c8d8f0;border-radius:8px;padding:.8rem 1.1rem;margin-bottom:1rem;font-size:.82rem;color:#1a2a50;line-height:1.6">Four research-grade tools: side-by-side treatment comparison, NNT/NNH extraction, structured data cards from any abstract, and a knowledge graph of article connections.</div>', unsafe_allow_html=True)

    adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
        "Treatment Comparison", "NNT / NNH Extractor",
        "Structured Extraction", "Knowledge Graph"
    ])

    # ── Sub-tab 1: Treatment Comparison ──────────────────────
    with adv_tab1:
        st.markdown('<div class="section-label">Treatment Comparison — A vs B</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.82rem;color:#6b7a8d;margin-bottom:1rem">Runs the full RAG + RL pipeline for two treatments and generates a head-to-head evidence comparison. Maps directly to how clinical decisions are made.</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,2,3])
        with c1: treat_a = st.text_input("Treatment A", placeholder="e.g. Metformin", key="treat_a")
        with c2: treat_b = st.text_input("Treatment B", placeholder="e.g. SGLT2 inhibitor", key="treat_b")
        with c3: condition = st.text_input("Condition / Clinical context", placeholder="e.g. type 2 diabetes glycemic control", key="treat_cond")
        run_compare = st.button("Compare treatments", key="run_compare")

        if run_compare and treat_a and treat_b and condition:
            from advanced_analysis import TreatmentComparator
            comparator = TreatmentComparator(groq_api_key=GROQ_API_KEY)
            pubmed_ret = None
            try:
                from pubmed_retriever import PubMedRetriever
                pubmed_ret = PubMedRetriever()
            except: pass

            with st.spinner(f"Retrieving evidence for {treat_a} and {treat_b}..."):
                result = comparator.compare(treat_a, treat_b, condition, pubmed_ret)

            st.divider()
            # Winner banner
            win_bg = "#f0faf5" if result.winner not in ("Comparable","") else "#f0f5ff"
            win_bc = "#b8e8d4" if result.winner not in ("Comparable","") else "#c8d8f0"
            win_tc = "#0a5c45" if result.winner not in ("Comparable","") else "#2d6fb3"
            st.markdown(f'''<div style="background:{win_bg};border:1.5px solid {win_bc};border-radius:10px;padding:1rem 1.3rem;margin-bottom:1rem">
    <div style="font-size:.72rem;font-weight:700;color:{win_tc};text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">Evidence verdict</div>
    <div style="font-size:1rem;font-weight:600;color:#0c1821">{"Winner: " + result.winner if result.winner not in ("Comparable","") else "Comparable evidence quality"}</div>
    <div style="font-size:.82rem;color:#6b7a8d;margin-top:3px">{result.winner_reason}</div>
</div>''', unsafe_allow_html=True)

            # Side-by-side grade metrics
            ca, cb = st.columns(2)
            with ca:
                st.metric(f"{treat_a} — avg grade", f"{result.avg_grade_a:.3f}")
                st.metric("Top grade", result.top_grade_a)
                st.metric("Articles found", len(result.articles_a))
            with cb:
                st.metric(f"{treat_b} — avg grade", f"{result.avg_grade_b:.3f}")
                st.metric("Top grade", result.top_grade_b)
                st.metric("Articles found", len(result.articles_b))

            # Head-to-head summary
            if result.head_to_head_summary:
                st.markdown(f'''<div class="summary-box" style="margin-top:1rem">
    <div class="summary-label">Head-to-head synthesis &nbsp;·&nbsp; {result.generated_by}</div>
    <div class="summary-text">{result.head_to_head_summary}</div>
</div>''', unsafe_allow_html=True)

            # Article lists side-by-side
            st.markdown('<div class="section-label" style="margin-top:1rem">Retrieved evidence</div>', unsafe_allow_html=True)
            la, lb = st.columns(2)
            def render_arts_col(arts, label, col):
                with col:
                    st.markdown(f'<div style="font-size:.78rem;font-weight:600;color:#1a2332;margin-bottom:.5rem">{label}</div>', unsafe_allow_html=True)
                    if not arts:
                        st.caption("No articles retrieved.")
                        return
                    for art in arts[:4]:
                        t = art.get("title","") if isinstance(art,dict) else getattr(art,"title","")
                        g = art.get("grade","C") if isinstance(art,dict) else getattr(art,"grade","C")
                        yr = art.get("year","") if isinstance(art,dict) else getattr(art,"pub_year","")
                        url = art.get("url","") if isinstance(art,dict) else getattr(art,"url","")
                        gcss = grade_css(str(g))
                        gl = grade_label(str(g))
                        pmid = art.get("pmid","") if isinstance(art,dict) else getattr(art,"pmid","")
                        pmid_link = f'<a href="{url}" target="_blank" style="color:#0a5c45;font-size:.68rem;font-family:IBM Plex Mono">PMID {pmid} ↗</a>' if url else ""
                        st.markdown(f'''<div style="background:#f8fafc;border:1px solid #dde3ee;border-radius:8px;padding:.7rem .9rem;margin-bottom:.5rem">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
        <span class="grade-pill {gcss}" style="font-size:.65rem">{g} — {gl}</span>
        <span style="font-size:.68rem;color:#6b7a8d;font-family:IBM Plex Mono">{yr}</span>
        <span style="margin-left:auto">{pmid_link}</span>
    </div>
    <div style="font-size:.8rem;color:#1a2332;line-height:1.4">{str(t)[:90]}{"..." if len(str(t))>90 else ""}</div>
</div>''', unsafe_allow_html=True)
            render_arts_col(result.articles_a, f"{treat_a} evidence", la)
            render_arts_col(result.articles_b, f"{treat_b} evidence", lb)

        elif run_compare:
            st.warning("Please fill in both treatments and the clinical condition.")
        else:
            st.markdown('''<div class="empty-state">
    <div class="empty-icon">⚖</div>
    <div class="empty-title">Enter two treatments and a condition</div>
    <div class="empty-sub">e.g. Metformin vs SGLT2 inhibitor for type 2 diabetes</div>
</div>''', unsafe_allow_html=True)

    # ── Sub-tab 2: NNT / NNH ─────────────────────────────────
    with adv_tab2:
        st.markdown('<div class="section-label">NNT / NNH Extractor</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.82rem;color:#6b7a8d;margin-bottom:1rem">Automatically extracts Number Needed to Treat, Number Needed to Harm, Absolute Risk Reduction and Relative Risk Reduction from article abstracts — the most clinically actionable statistics in evidence-based medicine.</div>', unsafe_allow_html=True)

        nnt_q = st.text_area("Clinical question to search", height=70, label_visibility="collapsed",
                              placeholder="e.g. statins for primary prevention of cardiovascular disease",
                              key="nnt_q")
        run_nnt = st.button("Extract NNT/NNH", key="run_nnt")

        if run_nnt and nnt_q.strip():
            from advanced_analysis import NNTExtractor
            extractor = NNTExtractor(groq_api_key=GROQ_API_KEY)
            pubmed_ret = None
            try:
                from pubmed_retriever import PubMedRetriever
                pubmed_ret = PubMedRetriever()
            except: pass

            with st.spinner("Searching PubMed and extracting statistics..."):
                articles = []
                if pubmed_ret:
                    try: articles = pubmed_ret.search(nnt_q, max_results=8)
                    except: pass
                nnt_results = extractor.extract_batch(articles, nnt_q)

            if nnt_results:
                st.markdown(f'<div style="font-size:.82rem;color:#6b7a8d;margin:.5rem 0">Found NNT/NNH/ARR data in {len(nnt_results)} of {len(articles)} retrieved articles.</div>', unsafe_allow_html=True)
                for r in nnt_results:
                    has_data = r.nnt is not None or r.nnh is not None or r.arr is not None
                    if not has_data: continue
                    st.markdown('<div style="background:#fff;border:1px solid #dde3ee;border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.8rem">', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.88rem;font-weight:600;color:#0c1821;margin-bottom:.5rem">{r.title}</div>', unsafe_allow_html=True)
                    cols = st.columns(4)
                    def stat_card(col, label, value, unit="", color="#0a5c45"):
                        with col:
                            if value is not None:
                                col.markdown(f'''<div style="background:#f0faf5;border:1px solid #b8e8d4;border-radius:8px;padding:.7rem;text-align:center">
    <div style="font-size:.65rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:.06em">{label}</div>
    <div style="font-size:1.5rem;font-weight:700;color:#0c1821;line-height:1.2">{value:.1f}<span style="font-size:.7rem;color:#6b7a8d">{unit}</span></div>
</div>''', unsafe_allow_html=True)
                            else:
                                col.markdown(f'<div style="background:#f8fafc;border:1px solid #e5e9f0;border-radius:8px;padding:.7rem;text-align:center"><div style="font-size:.65rem;color:#9aa5b4;text-transform:uppercase">{label}</div><div style="font-size:.9rem;color:#c8d0d8">—</div></div>', unsafe_allow_html=True)
                    stat_card(cols[0], "NNT", r.nnt)
                    stat_card(cols[1], "NNH", r.nnh, color="#9c2b2b")
                    stat_card(cols[2], "ARR", r.arr, "%")
                    stat_card(cols[3], "RRR", r.rrr, "%")
                    if r.interpretation:
                        st.markdown(f'<div style="background:#f0f5ff;border-left:3px solid #2d6fb3;border-radius:0 6px 6px 0;padding:7px 12px;margin-top:.6rem;font-size:.82rem;color:#1a2a40">{r.interpretation}</div>', unsafe_allow_html=True)
                    if r.outcome or r.timeframe or r.population:
                        details = " · ".join(filter(None, [r.population[:50] if r.population else "", r.timeframe, r.outcome[:50] if r.outcome else ""]))
                        st.caption(f"Context: {details}")
                    st.markdown('</div>', unsafe_allow_html=True)
            elif articles:
                st.info(f"Retrieved {len(articles)} articles but none contained explicit NNT/NNH values. Try a more specific clinical question or one focused on RCTs.")
            else:
                st.error("No articles retrieved. Check your NCBI API key or rephrase the query.")
        elif run_nnt:
            st.warning("Enter a clinical question first.")
        else:
            st.markdown('''<div class="empty-state">
    <div class="empty-icon">N</div>
    <div class="empty-title">Enter a clinical question</div>
    <div class="empty-sub">Works best with RCT-focused queries · e.g. "statins primary prevention cardiovascular"</div>
</div>''', unsafe_allow_html=True)

    # ── Sub-tab 3: Structured Extraction ─────────────────────
    with adv_tab3:
        st.markdown('<div class="section-label">Structured Data Extraction</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.82rem;color:#6b7a8d;margin-bottom:1rem">Converts any research abstract into a structured data card: study design, sample size, intervention, outcome, effect size, limitations. Useful for meta-analysis, systematic reviews, or any domain.</div>', unsafe_allow_html=True)

        se_q = st.text_area("Search query or paste abstract", height=80, label_visibility="collapsed",
                             placeholder="e.g. GLP-1 agonists weight loss obesity RCT",
                             key="se_q")
        run_se = st.button("Extract structured data", key="run_se")

        if run_se and se_q.strip():
            from advanced_analysis import StructuredExtractor
            extractor = StructuredExtractor(groq_api_key=GROQ_API_KEY, db_path=DB_PATH)
            pubmed_ret = None
            try:
                from pubmed_retriever import PubMedRetriever
                pubmed_ret = PubMedRetriever()
            except: pass

            with st.spinner("Retrieving and extracting structured data..."):
                articles = []
                if pubmed_ret:
                    try: articles = pubmed_ret.search(se_q, max_results=5)
                    except: pass
                if not articles and len(se_q) > 100:
                    # Treat input as a raw abstract
                    articles = [{"title": "Pasted abstract", "abstract": se_q, "pmid": "", "url": ""}]
                se_results = extractor.extract_batch(articles)

            for r in se_results:
                fields = [
                    ("Study design", r.study_design),
                    ("Sample size", r.sample_size),
                    ("Population", r.population),
                    ("Intervention", r.intervention),
                    ("Comparator", r.comparator),
                    ("Primary outcome", r.primary_outcome),
                    ("Key finding", r.key_finding),
                    ("Effect size", r.effect_size),
                    ("P-value", r.p_value),
                    ("Follow-up", r.follow_up),
                    ("Limitations", r.limitations),
                    ("Funding", r.funding),
                    ("Evidence grade", r.evidence_grade),
                ]
                filled = [(k,v) for k,v in fields if v and v not in ("Not reported","Not stated","")]
                if not filled: continue

                st.markdown(f'<div style="background:#fff;border:1px solid #dde3ee;border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.8rem">', unsafe_allow_html=True)
                if r.title:
                    pmid_link = f' <a href="https://pubmed.ncbi.nlm.nih.gov/{r.pmid}/" target="_blank" style="font-size:.68rem;color:#0a5c45;font-family:IBM Plex Mono">PMID {r.pmid} ↗</a>' if r.pmid else ""
                    st.markdown(f'<div style="font-size:.88rem;font-weight:600;color:#0c1821;margin-bottom:.8rem">{r.title}{pmid_link}</div>', unsafe_allow_html=True)

                # Render as 2-column grid
                left_fields = filled[:len(filled)//2+1]
                right_fields = filled[len(filled)//2+1:]
                cl, cr = st.columns(2)
                for col, flds in [(cl, left_fields), (cr, right_fields)]:
                    for label, val in flds:
                        col.markdown(f'''<div style="margin-bottom:.5rem">
    <div style="font-size:.65rem;font-weight:700;color:#6b7a8d;text-transform:uppercase;letter-spacing:.06em">{label}</div>
    <div style="font-size:.82rem;color:#1a2332;margin-top:1px">{val}</div>
</div>''', unsafe_allow_html=True)

                st.markdown(f'<div style="font-size:.68rem;color:#9aa5b4;margin-top:.5rem;font-family:IBM Plex Mono">Extracted by: {r.extracted_by}</div>', unsafe_allow_html=True)

                # JSON export button
                export_d = {f["Study design"[0] if f == fields[0] else k]: v
                            for k,v in filled}
                export_d = {k: v for k,v in filled}
                st.download_button("Download as JSON", data=json.dumps({k:v for k,v in filled}, indent=2),
                                   file_name=f"structured_{r.pmid or 'article'}.json",
                                   mime="application/json", key=f"dl_se_{r.pmid or hash(r.title)}")
                st.markdown('</div>', unsafe_allow_html=True)

        elif run_se:
            st.warning("Enter a query or paste an abstract.")
        else:
            st.markdown('''<div class="empty-state">
    <div class="empty-icon">{ }</div>
    <div class="empty-title">Search PubMed or paste an abstract</div>
    <div class="empty-sub">Works for any research domain · medical, CS, economics, social science</div>
</div>''', unsafe_allow_html=True)

    # ── Sub-tab 4: Knowledge Graph ────────────────────────────
    with adv_tab4:
        st.markdown('<div class="section-label">Knowledge Graph Visualiser</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.82rem;color:#6b7a8d;margin-bottom:1rem">Interactive network showing how retrieved articles connect via shared MeSH terms and co-authors. Node size = evidence grade. Node color = grade level. Hover to see article details.</div>', unsafe_allow_html=True)

        kg_q = st.text_area("Research topic to map", height=70, label_visibility="collapsed",
                             placeholder="e.g. SGLT2 inhibitors heart failure",
                             key="kg_q")
        run_kg = st.button("Build knowledge graph", key="run_kg")

        if run_kg and kg_q.strip():
            from advanced_analysis import KnowledgeGraphBuilder
            builder = KnowledgeGraphBuilder()
            pubmed_ret = None
            try:
                from pubmed_retriever import PubMedRetriever
                pubmed_ret = PubMedRetriever()
            except: pass

            with st.spinner("Retrieving articles and building graph..."):
                articles = []
                if pubmed_ret:
                    try: articles = pubmed_ret.search(kg_q, max_results=10)
                    except: pass
                graph = builder.build(articles)

            if graph.nodes:
                # Stats
                c1,c2,c3 = st.columns(3)
                c1.metric("Articles (nodes)", graph.stats.get("n_articles", 0))
                c2.metric("Connections (edges)", graph.stats.get("n_edges", 0))
                c3.metric("Topic clusters", graph.stats.get("n_clusters", 0))

                # Build Plotly figure
                try:
                    import plotly.graph_objects as go

                    # Edge traces
                    edge_x, edge_y = [], []
                    for e in graph.edges:
                        s = graph.nodes[e["source"]]
                        t = graph.nodes[e["target"]]
                        edge_x += [s["x"], t["x"], None]
                        edge_y += [s["y"], t["y"], None]

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=0.8, color="#dde3ee"),
                        hoverinfo="none", showlegend=False
                    )

                    # Node traces by grade
                    grade_groups = {}
                    for node in graph.nodes:
                        g = node["grade"]
                        if g not in grade_groups:
                            grade_groups[g] = {"x":[],"y":[],"text":[],"size":[],"color":node["color"],"grade":g}
                        grade_groups[g]["x"].append(node["x"])
                        grade_groups[g]["y"].append(node["y"])
                        grade_groups[g]["size"].append(node["size"])
                        hover = (f"<b>{node['full_title'][:60]}</b><br>"
                                 f"Grade: {node['grade']} · Year: {node['year']}<br>"
                                 f"PMID: {node['pmid']}<br>"
                                 + (f"MeSH: {', '.join(node['mesh'][:3])}" if node['mesh'] else ""))
                        grade_groups[g]["text"].append(hover)

                    grade_labels_map = {"A":"High (A)","H":"High","B":"Moderate (B)","M":"Moderate",
                                        "C":"Low (C)","L":"Low","D":"Very Low (D)","V":"Very Low"}
                    node_traces = []
                    for g, grp in sorted(grade_groups.items()):
                        node_traces.append(go.Scatter(
                            x=grp["x"], y=grp["y"], mode="markers",
                            marker=dict(size=grp["size"], color=grp["color"],
                                        line=dict(width=1.5, color="white")),
                            text=grp["text"], hoverinfo="text",
                            name=grade_labels_map.get(g, g),
                            showlegend=True
                        ))

                    fig = go.Figure(data=[edge_trace] + node_traces)
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(title="Evidence grade", font=dict(size=11)),
                        hovermode="closest",
                        margin=dict(b=20,l=5,r=5,t=20),
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        paper_bgcolor="white", plot_bgcolor="white",
                        height=480,
                        font=dict(family="IBM Plex Sans, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    # Plotly not available — show text-based network
                    st.info("Install plotly for interactive graph: pip install plotly")
                    for node in graph.nodes:
                        connected = [e for e in graph.edges if e["source"]==node["id"] or e["target"]==node["id"]]
                        st.markdown(f'<div style="background:#f8fafc;border:1px solid #dde3ee;border-radius:6px;padding:7px 11px;margin-bottom:4px"><span class="grade-pill {grade_css(node["grade"])}">{node["grade"]}</span> <span style="font-size:.82rem;color:#1a2332">{node["title"]}</span> <span style="font-size:.68rem;color:#6b7a8d;font-family:IBM Plex Mono">· {len(connected)} connections · {node["year"]}</span></div>', unsafe_allow_html=True)

                # Clusters
                if graph.clusters:
                    st.markdown('<div class="section-label" style="margin-top:1rem">Topic clusters</div>', unsafe_allow_html=True)
                    for term, indices in list(graph.clusters.items())[:6]:
                        titles = [graph.nodes[i]["title"] for i in indices if i < len(graph.nodes)]
                        st.markdown(f'''<div style="background:#f0f5ff;border:1px solid #c8d8f0;border-radius:8px;padding:.7rem 1rem;margin-bottom:.5rem">
    <div style="font-size:.75rem;font-weight:600;color:#2d6fb3;margin-bottom:3px">{term.title()}</div>
    <div style="font-size:.75rem;color:#6b7a8d">{" · ".join(titles[:3])}</div>
</div>''', unsafe_allow_html=True)
            else:
                st.error("No articles retrieved or insufficient metadata for graph. Try a broader topic or check your NCBI API key.")

        elif run_kg:
            st.warning("Enter a research topic first.")
        else:
            st.markdown('''<div class="empty-state">
    <div class="empty-icon">o</div>
    <div class="empty-title">Enter a research topic</div>
    <div class="empty-sub">Connects articles via shared MeSH terms and co-authors · works for any research field</div>
</div>''', unsafe_allow_html=True)