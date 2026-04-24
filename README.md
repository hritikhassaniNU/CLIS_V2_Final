# CLIS V2 — Clinical Literature Intelligence System

> **One app. One command: `streamlit run app.py`**

**Course:** INFO 7375 — Generative AI & Prompt Engineering
**Student:** Hritik Hassani | Northeastern University | Spring 2026

---

## What is this?

CLIS V2 is a clinical decision support system that helps physicians find and evaluate medical evidence. It combines three AI techniques:

- **RAG** — retrieves real papers from PubMed in real time (no hallucinated citations)
- **Reinforcement Learning** — UCB Bandit optimizes the search strategy, REINFORCE agent ranks retrieved evidence
- **GRADE methodology** — rates evidence quality (RCT > cohort > case report)

It also includes an **ICD-10 coding assistant** (RAG over 47 CMS guideline sections) and an **automated benchmark suite** (10 test cases including a hallucination trap).

---

## Project structure

```
CLIS_V2/
│
├── app.py                       <- ENTRY POINT — run this
│
├── tools/
│   ├── pubmed_retriever.py          Live NCBI PubMed API
│   ├── grade_evidence_tool.py       GRADE evidence grading
│   ├── icd10_rag_engine.py          ICD-10 RAG (47 CMS sections)
│   ├── citation_grounder.py         Paragraph-level citation grounding
│   ├── benchmark_evaluator.py       10 automated test cases
│   └── persistent_bandit.py         SQLite-backed RLHF bandit
│
├── models/
│   ├── bandit_policy.pkl            Trained UCB bandit
│   └── reinforce_policy.pkl         Trained REINFORCE policy
│
├── notebooks/
│   ├── NB1_ContextualBandit.ipynb
│   ├── NB2_REINFORCE.ipynb
│   ├── NB3_Integration.ipynb
│   ├── NB3b_LivePubMed.ipynb
│   ├── NB4_StatisticalValidation.ipynb
│   ├── NB5_GradeTool.ipynb
│   └── NB6_AblationStudy.ipynb
│
├── results/                     8 experiment charts + grade_assessments.json
├── docs/                        Technical report PDF + project proposal
├── requirements.txt
└── .env.example
```

---

## Setup

### Step 1 — Get the code

```bash
git clone https://github.com/YOUR_USERNAME/CLIS_V2.git
cd CLIS_V2
```

### Step 2 — Create virtual environment

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

No GPU required. Everything runs on CPU.

### Step 4 — API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_key_here
NCBI_API_KEY=your_key_here
```

| Key | Where | Why |
|---|---|---|
| GROQ_API_KEY | console.groq.com (free) | LLM summaries, GRADE grading, contradiction detection |
| NCBI_API_KEY | ncbi.nlm.nih.gov/account (free) | PubMed: 3 req/sec without key, 10 req/sec with |

The app runs without keys in rule-based fallback mode. Both keys are free, no credit card.

### Step 5 — Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## The 4 tabs

| Tab | What it does |
|---|---|
| Clinical search | Question -> PubMed -> GRADE -> REINFORCE -> LLM summary with citation grounding -> contradiction check -> download report |
| ICD-10 coding | Coding question -> RAG over 47 CMS sections -> structured answer with codes and sequencing rules |
| Evaluation dashboard | Run all 10 benchmark tests including hallucination trap (TC10: fake trial that doesn't exist) |
| System analytics | Live bandit learning curve, RLHF feedback stats, arm performance per context |

---

## RLHF — thumbs up/down

After every search, click Helpful or Not helpful. This updates the bandit's reward estimate in real time and persists across sessions via SQLite. The system gets smarter the more it is used.

---

## Key results

| Metric | Value |
|---|---|
| UCB bandit improvement | +4.01% over random (p=0.0048, Cohen's d=2.735) |
| REINFORCE policy loss reduction | 72.6% +/- 5.7% across 5 seeds |
| Arm identification accuracy | 100% all seeds |
| ICD-10 coverage | 47 CMS guideline sections |
| Benchmark suite | 10 tests (3 critical + hallucination trap) |

---

## Tech stack — $0 total cost

| Component | Technology |
|---|---|
| LLM | Groq — Llama 3.3 70B (free tier) |
| Literature | PubMed NCBI E-utilities (free) |
| ICD-10 RAG | Custom TF-IDF, no external deps |
| RL | PyTorch — custom UCB + REINFORCE |
| Persistence | SQLite (Python stdlib) |
| UI | Streamlit |

---

*Hritik Hassani · Northeastern University · INFO 7375 · Spring 2026*
