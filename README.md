# CLIS V2 — Clinical Literature Intelligence System

> **Course:** INFO 7375 — Generative AI & Prompt Engineering
> **Student:** Hritik Hassani | Northeastern University | Spring 2026

---

## Links

| | |
|---|---|
| **GitHub Repo** | [https://github.com/hritikhassaniNU/CLIS_V2_Final](https://github.com/hritikhassaniNU/CLIS_V2_Final) |
| **Live App** | [https://clisv2final-4qi6fpsbtgg2qsf8wwn8ci.streamlit.app/](https://clisv2final-4qi6fpsbtgg2qsf8wwn8ci.streamlit.app/) |
| **Demo Video** | [https://drive.google.com/drive/folders/1_71V_uYAU17P2Tc3l7cdwfVwx9znJQpm?usp=sharing](https://drive.google.com/drive/folders/1_71V_uYAU17P2Tc3l7cdwfVwx9znJQpm?usp=sharing) |

---

## Live Demo

**Try it now — no setup required:**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clisv2final-4qi6fpsbtgg2qsf8wwn8ci.streamlit.app/)

> The deployed app runs on Streamlit Community Cloud. Open the link, wait ~10 seconds for cold start, and start searching.

---

## What is CLIS V2?

CLIS V2 is a clinical decision support system that helps physicians find, evaluate, and interpret medical evidence from PubMed — in real time, with hallucination prevention built in.

It solves two problems:

1. **PubMed is too slow** — a proper evidence search takes 16+ minutes. CLIS V2 does it in seconds.
2. **General AI is unsafe** — ChatGPT and Gemini confidently fabricate studies. CLIS V2 grounds every sentence in a real source and flags anything it can't verify.

**Three AI techniques combined:**

- **Dual RAG** — live PubMed retrieval + ICD-10 CMS guidelines (47 sections)
- **Reinforcement Learning** — UCB Contextual Bandit (query strategy) + REINFORCE policy gradient (article ranking) + RLHF feedback loop
- **Hallucination Prevention** — per-sentence Jaccard citation grounding + hallucination trap (TC10) in benchmark suite

---

## Option A — Use the Deployed App (Recommended)

No installation needed.

1. Open: [https://clisv2final-4qi6fpsbtgg2qsf8wwn8ci.streamlit.app/](https://clisv2final-4qi6fpsbtgg2qsf8wwn8ci.streamlit.app/)
2. Wait for the app to load (~10–15 seconds on cold start)
3. The sidebar shows system status — RL models, Groq connection, ICD-10 RAG

> **Note:** The deployed app runs without a Groq API key by default — summaries use rule-based fallback mode. To see full LLM output, run locally with your own key (see Option B).

---

## Option B — Run Locally

### Step 1 — Clone the repo

```bash
git clone https://github.com/hritikhassaniNU/CLIS_V2_Final.git
cd CLIS_V2_Final
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

### Step 4 — Set up API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_key_here
NCBI_API_KEY=your_key_here
```

| Key | Where to get it | What it does |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free | LLM summaries, GRADE grading, contradiction detection |
| `NCBI_API_KEY` | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) — free | PubMed: 10 req/sec with key vs 3 req/sec without |

> Both keys are free. No credit card required. The app runs without them in rule-based fallback mode — but Groq is required for full LLM output.

### Step 5 — Run

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501)

---

## The 5 Tabs

| Tab | What it does |
|---|---|
| **Clinical Search** | Type a clinical question → PubMed retrieval → GRADE evidence grading → REINFORCE ranking → Groq LLM summary → citation grounding → contradiction detection |
| **ICD-10 Coding** | Type a coding question → RAG over 47 CMS guideline sections → structured answer with primary codes, sequencing rules, and combination coding notes |
| **Advanced Analysis** | Treatment comparison (head-to-head evidence), NNT/NNH extractor, structured abstract extraction (13 fields, JSON export) |
| **Benchmark Evaluation** | Run all 10 automated test cases — including TC10, the hallucination trap (queries a trial that doesn't exist) |
| **System Analytics** | UCB bandit learning curve, arm reward matrix, RLHF feedback counts, REINFORCE policy loss history |

---

## How to Use — Quick Start

### Clinical Search (Tab 1)

**Free Text mode:**
1. Type your clinical question in the box
2. Click **Run Pipeline**
3. Watch the 5-step pipeline execute: Context Classification → UCB Bandit → PubMed Retrieval → GRADE Grading → REINFORCE Ranking
4. Read the LLM summary, citation grounding status, and contradiction check below

**PICO Builder mode:**
1. Switch to PICO Builder at the top
2. Fill in Population, Intervention, Comparison (optional), Outcome
3. The system builds the structured clinical question automatically
4. Click **Run Pipeline**

**After each result:**
- Click **Helpful** or **Not helpful** to update the bandit's reward estimates via RLHF
- Click any PMID link to open the original article on PubMed
- Download the evidence report as TXT, MD, or JSON

---

### ICD-10 Coding (Tab 2)

1. Type a coding question (e.g. *"How do I code type 2 diabetes with CKD stage 3?"*)
2. Click **Look up code** — or click one of the quick lookup buttons
3. The system retrieves the matching CMS guideline sections and generates a structured coding answer

---

### Benchmark (Tab 4)

1. Click **Run all 10 tests**
2. Wait ~60 seconds — tests run against live PubMed and Groq
3. Check TC10 (hallucination trap) — system should return nothing or flag all output as UNGROUNDED

---

## RLHF — Making the System Smarter

After every clinical search result, click **Helpful** or **Not helpful**.

Each click updates the bandit's mean reward estimate by ±0.15 for the arm that was used, for the context that was detected. The update persists to SQLite immediately — so the next search benefits from your feedback even after the app restarts.

The more it is used, the better the query strategy selection becomes.

---

## Project Structure

```
CLIS_V2/
│
├── app.py                           ← ENTRY POINT — run this
│
├── tools/
│   ├── pubmed_retriever.py              Live NCBI PubMed API
│   ├── grade_evidence_tool.py           GRADE evidence grading
│   ├── icd10_rag_engine.py              ICD-10 RAG (47 CMS sections)
│   ├── citation_grounder.py             Per-sentence citation grounding
│   ├── benchmark_evaluator.py           10 automated test cases
│   └── persistent_bandit.py             SQLite-backed RLHF bandit
│
├── models/
│   ├── bandit_policy.pkl                Trained UCB bandit
│   └── reinforce_policy.pkl             Trained REINFORCE policy
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
├── results/                         8 experiment charts + grade_assessments.json
├── docs/                            Technical report PDF + architecture doc
├── requirements.txt
└── .env.example
```

---

## Key Results

| Metric | Value |
|---|---|
| UCB bandit improvement | +4.01% over random (p = 0.0048, Cohen's d = 2.735) |
| REINFORCE policy loss reduction | 72.6% ± 5.7% across 5 seeds × 300 episodes |
| Benchmark suite | 10/10 passing including hallucination trap |
| ICD-10 coverage | 47 CMS FY2024 guideline sections |
| Infrastructure cost | $0 — entirely free tier |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — Llama 3.3 70B (free tier) |
| Literature retrieval | PubMed NCBI E-utilities (free) |
| ICD-10 RAG | Custom TF-IDF, no external dependencies |
| Reinforcement Learning | PyTorch — custom UCB Bandit + REINFORCE |
| Persistence | SQLite (Python stdlib) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud (free) |

---

## Troubleshooting

**App loads but summaries say "Template" instead of showing LLM output:**
→ Groq API key is not set. Add `GROQ_API_KEY` to your `.env` file (local) or Streamlit secrets (deployed).

**PubMed returns no results:**
→ The app falls back to simulated data automatically and shows a warning. This is normal on slow or rate-limited connections. Add `NCBI_API_KEY` to increase rate limit.

**`torch` install fails or app crashes on startup:**
→ Use the CPU-only PyTorch build. Add to `requirements.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.0+cpu
```

**Streamlit Cloud deploy fails with memory error:**
→ Switch to CPU-only torch (see above). PyTorch full build is ~800MB and can exceed the 1GB Streamlit Cloud limit.

---

*Hritik Hassani · Northeastern University · INFO 7375 · Spring 2026*