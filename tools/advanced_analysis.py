"""
CLIS V2 — Advanced Analysis Tools
====================================
Student: Hritik Ram | Northeastern University

Four new tools:
  1. TreatmentComparator  — A vs B side-by-side evidence comparison
  2. NNTExtractor          — Number Needed to Treat / NNH from abstracts
  3. StructuredExtractor   — Abstract → structured JSON data card
  4. KnowledgeGraphBuilder — Article network from MeSH + author metadata
"""

import os, re, json, math, hashlib, time, sqlite3
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════
# 1. TREATMENT COMPARATOR
# ══════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    treatment_a: str = ""
    treatment_b: str = ""
    articles_a: list = field(default_factory=list)
    articles_b: list = field(default_factory=list)
    avg_grade_a: float = 0.0
    avg_grade_b: float = 0.0
    top_grade_a: str = "C"
    top_grade_b: str = "C"
    winner: str = ""
    winner_reason: str = ""
    head_to_head_summary: str = ""
    a_summary: str = ""
    b_summary: str = ""
    generated_by: str = "Rule-based"


class TreatmentComparator:
    """
    Runs the full RAG + RL pipeline for two treatments
    and synthesises a head-to-head evidence comparison.
    """

    GRADE_VALS = {"A": 1.0, "H": 1.0, "B": 0.75, "M": 0.75,
                  "C": 0.45, "L": 0.45, "D": 0.15, "V": 0.15}

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        if self.api_key and self.api_key not in ("", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass

    def compare(self, treatment_a: str, treatment_b: str,
                condition: str, pubmed_retriever=None) -> ComparisonResult:
        """
        Compare evidence for treatment_a vs treatment_b for a given condition.
        """
        result = ComparisonResult(treatment_a=treatment_a, treatment_b=treatment_b)

        # Build queries
        q_a = f"{treatment_a} {condition} efficacy outcomes"
        q_b = f"{treatment_b} {condition} efficacy outcomes"

        # Retrieve articles for each
        arts_a = self._retrieve(q_a, pubmed_retriever)
        arts_b = self._retrieve(q_b, pubmed_retriever)

        result.articles_a = arts_a[:5]
        result.articles_b = arts_b[:5]

        # Compute grade summaries
        result.avg_grade_a = self._avg_grade(arts_a)
        result.avg_grade_b = self._avg_grade(arts_b)
        result.top_grade_a = self._top_grade(arts_a)
        result.top_grade_b = self._top_grade(arts_b)

        # Determine winner by evidence grade
        if result.avg_grade_a > result.avg_grade_b + 0.05:
            result.winner = treatment_a
            result.winner_reason = f"Higher average evidence grade ({result.avg_grade_a:.2f} vs {result.avg_grade_b:.2f})"
        elif result.avg_grade_b > result.avg_grade_a + 0.05:
            result.winner = treatment_b
            result.winner_reason = f"Higher average evidence grade ({result.avg_grade_b:.2f} vs {result.avg_grade_a:.2f})"
        else:
            result.winner = "Comparable"
            result.winner_reason = f"Similar evidence quality ({result.avg_grade_a:.2f} vs {result.avg_grade_b:.2f})"

        # LLM head-to-head synthesis
        if self.client and arts_a and arts_b:
            result.head_to_head_summary, result.a_summary, result.b_summary, result.generated_by = \
                self._synthesize(treatment_a, treatment_b, condition, arts_a[:3], arts_b[:3])
        else:
            result.head_to_head_summary = self._rule_based_comparison(treatment_a, treatment_b, result)
            result.generated_by = "Rule-based"

        return result

    def _retrieve(self, query: str, retriever) -> list:
        if retriever is None:
            return []
        try:
            articles = retriever.search(query, max_results=5)
            return articles if articles else []
        except Exception:
            return []

    def _avg_grade(self, articles: list) -> float:
        if not articles:
            return 0.4
        vals = [self.GRADE_VALS.get(
            str(a.get("grade", "C") if isinstance(a, dict) else getattr(a, "grade", "C"))[0], 0.4
        ) for a in articles]
        return round(sum(vals) / len(vals), 3)

    def _top_grade(self, articles: list) -> str:
        if not articles:
            return "C"
        grades = [str(a.get("grade", "C") if isinstance(a, dict) else getattr(a, "grade", "C"))[0]
                  for a in articles]
        order = ["A", "H", "B", "M", "C", "L", "D", "V"]
        for g in order:
            if g in grades:
                return g
        return "C"

    def _synthesize(self, ta, tb, condition, arts_a, arts_b):
        def art_text(arts, name):
            lines = []
            for a in arts:
                t = a.get("title", "") if isinstance(a, dict) else getattr(a, "title", "")
                g = a.get("grade", "C") if isinstance(a, dict) else getattr(a, "grade", "C")
                yr = a.get("year", "") if isinstance(a, dict) else getattr(a, "pub_year", "")
                ab = str(a.get("abstract", "") if isinstance(a, dict) else getattr(a, "abstract", ""))[:200]
                lines.append(f"[{g}] {t} ({yr}): {ab}")
            return f"{name} evidence:\n" + "\n".join(lines)

        context = art_text(arts_a, ta) + "\n\n" + art_text(arts_b, tb)

        prompt = f"""You are a clinical evidence analyst comparing two treatments.

Condition: {condition}
Treatment A: {ta}
Treatment B: {tb}

{context}

Provide a structured comparison with exactly these sections:
1. {ta} summary (2 sentences: main finding + evidence quality)
2. {tb} summary (2 sentences: main finding + evidence quality)
3. Head-to-head verdict (2 sentences: which has stronger evidence and why, with clinical implication)

Be clinically precise. Reference specific PMIDs where available."""

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15, max_tokens=400
            )
            full = resp.choices[0].message.content.strip()
            # Parse sections
            lines = full.split("\n")
            a_sum = " ".join(l for l in lines if ta.lower()[:8] in l.lower() or "1." in l)[:300]
            b_sum = " ".join(l for l in lines if tb.lower()[:8] in l.lower() or "2." in l)[:300]
            verdict = " ".join(l for l in lines if "verdict" in l.lower() or "3." in l or "head" in l.lower())[:300]
            if not verdict:
                verdict = lines[-1] if lines else full[-200:]
            return verdict or full[:300], a_sum or full[:150], b_sum or full[150:300], "Groq Llama 3.3 70B"
        except Exception as e:
            return self._rule_based_comparison(ta, tb, None), "", "", f"Rule-based (error: {str(e)[:40]})"

    def _rule_based_comparison(self, ta, tb, result) -> str:
        if result is None:
            return f"Comparison of {ta} vs {tb} requires evidence retrieval."
        if result.winner == "Comparable":
            return (f"Evidence for {ta} (avg grade {result.avg_grade_a:.2f}) and {tb} "
                    f"(avg grade {result.avg_grade_b:.2f}) is comparable in quality for this condition. "
                    f"Clinical context, patient preference, and cost should guide treatment selection.")
        return (f"Based on retrieved evidence, {result.winner} has stronger evidence quality "
                f"({result.winner_reason}). Physician review required before treatment decisions.")


# ══════════════════════════════════════════════════════════════
# 2. NNT / NNH EXTRACTOR
# ══════════════════════════════════════════════════════════════

@dataclass
class NNTResult:
    title: str = ""
    pmid: str = ""
    nnt: Optional[float] = None
    nnh: Optional[float] = None
    arr: Optional[float] = None
    rrr: Optional[float] = None
    rer: Optional[float] = None
    outcome: str = ""
    timeframe: str = ""
    population: str = ""
    confidence_interval: str = ""
    interpretation: str = ""
    extracted_by: str = ""
    raw_stats: dict = field(default_factory=dict)


class NNTExtractor:
    """
    Extracts Number Needed to Treat (NNT), Number Needed to Harm (NNH),
    Absolute Risk Reduction (ARR), and Relative Risk Reduction (RRR)
    from article abstracts using Groq LLM + regex fallback.

    NNT is the most clinically actionable statistic in medicine —
    "treat N patients to prevent 1 event" is immediately usable.
    """

    # Regex patterns for common statistical formats
    NNT_PATTERNS = [
        r'nnt\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)',
        r'number needed to treat\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*patients.*?needed.*?treat',
    ]
    NNH_PATTERNS = [
        r'nnh\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)',
        r'number needed to harm\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)',
    ]
    ARR_PATTERNS = [
        r'arr\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'absolute risk reduction\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)\s*%?',
    ]
    RRR_PATTERNS = [
        r'rrr\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'relative risk reduction\s*(?:of|=|was|is)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:relative\s*)?reduction',
    ]

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        if self.api_key and self.api_key not in ("", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass

    def extract_batch(self, articles: list, question: str) -> list:
        """Extract NNT/NNH from a list of articles. Returns list of NNTResult."""
        results = []
        for art in articles:
            abstract = str(art.get("abstract", "") if isinstance(art, dict) else getattr(art, "abstract", ""))
            title = str(art.get("title", "") if isinstance(art, dict) else getattr(art, "title", ""))
            pmid = str(art.get("pmid", "") if isinstance(art, dict) else getattr(art, "pmid", ""))
            if abstract and len(abstract) > 50:
                r = self.extract(abstract, title, pmid, question)
                if r.nnt is not None or r.nnh is not None or r.arr is not None:
                    results.append(r)
        return results

    def extract(self, abstract: str, title: str = "", pmid: str = "", question: str = "") -> NNTResult:
        result = NNTResult(title=title[:80], pmid=pmid)

        # Try regex first (fast, no API call)
        result = self._regex_extract(abstract, result)

        # If regex found nothing meaningful, try LLM
        if self.client and result.nnt is None and result.nnh is None:
            result = self._llm_extract(abstract, title, question, result)

        # Build interpretation
        if result.nnt is not None:
            result.interpretation = (
                f"You need to treat {result.nnt:.0f} patients to prevent 1 {result.outcome or 'adverse event'}. "
                f"{'Lower NNT = more effective treatment.' if result.nnt < 20 else 'Higher NNT suggests modest benefit.'}"
            )
        elif result.nnh is not None:
            result.interpretation = (
                f"For every {result.nnh:.0f} patients treated, 1 additional harm occurs. "
                f"Compare with NNT to assess benefit-harm balance."
            )
        elif result.arr is not None:
            nnt_calc = round(100 / result.arr) if result.arr > 0 else None
            result.interpretation = (
                f"Absolute risk reduction of {result.arr:.1f}%. "
                f"{'Estimated NNT: ~' + str(nnt_calc) if nnt_calc else ''}"
            )

        return result

    def _regex_extract(self, text: str, result: NNTResult) -> NNTResult:
        t = text.lower()
        for p in self.NNT_PATTERNS:
            m = re.search(p, t)
            if m:
                try:
                    result.nnt = float(m.group(1))
                    result.extracted_by = "regex"
                    break
                except (ValueError, IndexError):
                    pass
        for p in self.NNH_PATTERNS:
            m = re.search(p, t)
            if m:
                try:
                    result.nnh = float(m.group(1))
                    result.extracted_by = "regex"
                    break
                except (ValueError, IndexError):
                    pass
        for p in self.ARR_PATTERNS:
            m = re.search(p, t)
            if m:
                try:
                    result.arr = float(m.group(1))
                    result.extracted_by = "regex"
                    break
                except (ValueError, IndexError):
                    pass
        for p in self.RRR_PATTERNS:
            m = re.search(p, t)
            if m:
                try:
                    result.rrr = float(m.group(1))
                    result.extracted_by = result.extracted_by or "regex"
                    break
                except (ValueError, IndexError):
                    pass
        return result

    def _llm_extract(self, abstract: str, title: str, question: str, result: NNTResult) -> NNTResult:
        prompt = f"""Extract clinical statistics from this abstract. Return ONLY valid JSON.

Abstract: {abstract[:600]}

Extract these if present (null if not stated):
{{
  "nnt": number or null,
  "nnh": number or null,
  "arr": number or null (as percentage, e.g. 3.5 for 3.5%),
  "rrr": number or null (as percentage),
  "outcome": "primary outcome measured",
  "timeframe": "follow-up duration",
  "population": "patient population",
  "confidence_interval": "CI if stated"
}}

Only extract numbers explicitly stated. Do not calculate or estimate."""
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200
            )
            raw = re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip()
            d = json.loads(raw)
            if d.get("nnt") is not None:
                result.nnt = float(d["nnt"])
            if d.get("nnh") is not None:
                result.nnh = float(d["nnh"])
            if d.get("arr") is not None:
                result.arr = float(d["arr"])
            if d.get("rrr") is not None:
                result.rrr = float(d["rrr"])
            result.outcome = str(d.get("outcome", ""))[:100]
            result.timeframe = str(d.get("timeframe", ""))[:60]
            result.population = str(d.get("population", ""))[:100]
            result.confidence_interval = str(d.get("confidence_interval", ""))[:80]
            result.raw_stats = d
            result.extracted_by = "Groq Llama 3.3 70B"
        except Exception:
            result.extracted_by = "Not found"
        return result


# ══════════════════════════════════════════════════════════════
# 3. STRUCTURED DATA EXTRACTOR
# ══════════════════════════════════════════════════════════════

@dataclass
class StructuredArticle:
    title: str = ""
    pmid: str = ""
    study_design: str = ""
    sample_size: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    primary_outcome: str = ""
    key_finding: str = ""
    effect_size: str = ""
    p_value: str = ""
    follow_up: str = ""
    limitations: str = ""
    funding: str = ""
    evidence_grade: str = ""
    extracted_by: str = ""


class StructuredExtractor:
    """
    Converts any article abstract into a structured data card (JSON).
    Useful for meta-analysis, systematic reviews, or any domain comparison.
    Works on medical AND non-medical abstracts.
    """

    def __init__(self, groq_api_key: Optional[str] = None, db_path: str = "clis_data.db"):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.db_path = db_path
        self._init_cache()
        if self.api_key and self.api_key not in ("", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass

    def _init_cache(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""CREATE TABLE IF NOT EXISTS struct_cache
                (key TEXT PRIMARY KEY, result TEXT, ts REAL)""")
            conn.commit(); conn.close()
        except Exception:
            pass

    def _cache_get(self, key):
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute("SELECT result FROM struct_cache WHERE key=?", (key,)).fetchone()
            conn.close()
            return json.loads(row[0]) if row else None
        except Exception:
            return None

    def _cache_set(self, key, val):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT OR REPLACE INTO struct_cache VALUES(?,?,?)",
                         (key, json.dumps(val), time.time()))
            conn.commit(); conn.close()
        except Exception:
            pass

    def extract(self, abstract: str, title: str = "", pmid: str = "") -> StructuredArticle:
        cache_key = hashlib.md5((abstract[:200] + title[:50]).encode()).hexdigest()
        cached = self._cache_get(cache_key)
        if cached:
            return StructuredArticle(**{k: v for k, v in cached.items()
                                        if k in StructuredArticle.__dataclass_fields__})

        result = StructuredArticle(title=title[:80], pmid=pmid)

        if self.client:
            result = self._llm_extract(abstract, title, result)
        else:
            result = self._rule_extract(abstract, result)

        self._cache_set(cache_key, {
            "title": result.title, "pmid": result.pmid,
            "study_design": result.study_design, "sample_size": result.sample_size,
            "population": result.population, "intervention": result.intervention,
            "comparator": result.comparator, "primary_outcome": result.primary_outcome,
            "key_finding": result.key_finding, "effect_size": result.effect_size,
            "p_value": result.p_value, "follow_up": result.follow_up,
            "limitations": result.limitations, "funding": result.funding,
            "evidence_grade": result.evidence_grade, "extracted_by": result.extracted_by,
        })
        return result

    def extract_batch(self, articles: list) -> list:
        results = []
        for art in articles:
            abstract = str(art.get("abstract", "") if isinstance(art, dict) else getattr(art, "abstract", ""))
            title = str(art.get("title", "") if isinstance(art, dict) else getattr(art, "title", ""))
            pmid = str(art.get("pmid", "") if isinstance(art, dict) else getattr(art, "pmid", ""))
            if abstract and len(abstract) > 30:
                results.append(self.extract(abstract, title, pmid))
        return results

    def _llm_extract(self, abstract: str, title: str, result: StructuredArticle) -> StructuredArticle:
        prompt = f"""Extract structured data from this research abstract. Return ONLY valid JSON.

Title: {title}
Abstract: {abstract[:700]}

{{
  "study_design": "RCT/cohort/meta-analysis/case-control/review/other",
  "sample_size": "n=X or Not reported",
  "population": "who was studied",
  "intervention": "what intervention/exposure",
  "comparator": "comparison group or None",
  "primary_outcome": "main outcome measured",
  "key_finding": "1 sentence main result",
  "effect_size": "OR/RR/HR/MD with value or Not reported",
  "p_value": "p-value if stated or Not reported",
  "follow_up": "duration or Not reported",
  "limitations": "key limitation in 1 sentence",
  "funding": "funding source or Not stated",
  "evidence_grade": "HIGH/MODERATE/LOW/VERY_LOW"
}}"""
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05, max_tokens=400
            )
            raw = re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip()
            d = json.loads(raw)
            for field_name in ["study_design", "sample_size", "population", "intervention",
                                "comparator", "primary_outcome", "key_finding", "effect_size",
                                "p_value", "follow_up", "limitations", "funding", "evidence_grade"]:
                val = d.get(field_name, "")
                if val and str(val).lower() not in ("null", "none", "n/a", "not reported", "not stated"):
                    setattr(result, field_name, str(val)[:150])
            result.extracted_by = "Groq Llama 3.3 70B"
        except Exception as e:
            result = self._rule_extract(abstract, result)
        return result

    def _rule_extract(self, abstract: str, result: StructuredArticle) -> StructuredArticle:
        t = abstract.lower()
        if any(p in t for p in ["randomized", "randomised", "rct", "placebo"]):
            result.study_design = "RCT"
        elif any(p in t for p in ["systematic review", "meta-analysis"]):
            result.study_design = "Systematic review / meta-analysis"
        elif any(p in t for p in ["cohort", "prospective", "longitudinal"]):
            result.study_design = "Cohort study"
        elif any(p in t for p in ["case-control", "case control"]):
            result.study_design = "Case-control"
        else:
            result.study_design = "Observational / review"
        n_match = re.search(r"n\s*=\s*(\d[\d,]+)", t) or re.search(r"(\d[\d,]+)\s+patients", t)
        if n_match:
            result.sample_size = f"n={n_match.group(1)}"
        result.extracted_by = "Rule-based"
        return result


# ══════════════════════════════════════════════════════════════
# 4. KNOWLEDGE GRAPH BUILDER
# ══════════════════════════════════════════════════════════════

@dataclass
class GraphData:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    clusters: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


class KnowledgeGraphBuilder:
    """
    Builds an article knowledge graph from PubMed metadata.

    Nodes: articles
    Edges: shared MeSH terms, shared authors, year proximity
    Clusters: MeSH term groups

    Returns data formatted for Plotly networkx visualisation.
    Works on any domain — not just medical literature.
    """

    def build(self, articles: list) -> GraphData:
        if not articles or len(articles) < 2:
            return GraphData()

        nodes = []
        edges = []
        mesh_index = {}   # term → [article indices]
        author_index = {}  # author → [article indices]

        # Build nodes
        for i, art in enumerate(articles):
            title = str(art.get("title", "") if isinstance(art, dict) else getattr(art, "title", ""))
            pmid  = str(art.get("pmid", "") if isinstance(art, dict) else getattr(art, "pmid", ""))
            grade = str(art.get("grade", "C") if isinstance(art, dict) else getattr(art, "grade", "C"))
            year  = str(art.get("year", "") if isinstance(art, dict) else getattr(art, "pub_year", ""))
            url   = str(art.get("url", "") if isinstance(art, dict) else getattr(art, "url", ""))
            mesh  = list(art.get("mesh_terms", []) if isinstance(art, dict) else getattr(art, "mesh_terms", []))
            authors = list(art.get("authors", []) if isinstance(art, dict) else getattr(art, "authors", []))

            grade_color = {"A": "#0a5c45", "H": "#0a5c45",
                           "B": "#92600a", "M": "#92600a",
                           "C": "#9c2b2b", "L": "#9c2b2b",
                           "D": "#5a5a5a"}.get(grade[0] if grade else "C", "#6b7a8d")

            nodes.append({
                "id": i, "pmid": pmid, "url": url,
                "title": title[:60] + ("..." if len(title) > 60 else ""),
                "full_title": title,
                "grade": grade[0] if grade else "C",
                "year": year,
                "color": grade_color,
                "size": 14 + ({"A": 10, "H": 10, "B": 6, "M": 6, "C": 2, "L": 2, "D": 0}.get(grade[0] if grade else "C", 2)),
                "mesh": mesh[:5],
                "authors": authors[:3],
            })

            for term in mesh[:8]:
                t = term.lower().strip()
                if t not in mesh_index:
                    mesh_index[t] = []
                mesh_index[t].append(i)

            for author in authors[:3]:
                a = author.lower().strip()
                if a not in author_index:
                    author_index[a] = []
                author_index[a].append(i)

        # Build edges from shared MeSH terms
        edge_set = set()
        for term, indices in mesh_index.items():
            if len(indices) >= 2:
                for x in range(len(indices)):
                    for y in range(x + 1, len(indices)):
                        key = (min(indices[x], indices[y]), max(indices[x], indices[y]))
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append({
                                "source": key[0], "target": key[1],
                                "weight": 1.0, "reason": f"Shared MeSH: {term[:30]}",
                                "type": "mesh"
                            })

        # Build edges from shared authors
        for author, indices in author_index.items():
            if len(indices) >= 2:
                for x in range(len(indices)):
                    for y in range(x + 1, len(indices)):
                        key = (min(indices[x], indices[y]), max(indices[x], indices[y]))
                        # Increase weight if already connected
                        existing = next((e for e in edges if e["source"] == key[0] and e["target"] == key[1]), None)
                        if existing:
                            existing["weight"] = min(existing["weight"] + 0.5, 3.0)
                            existing["reason"] += f" + shared author"
                        elif key not in edge_set:
                            edge_set.add(key)
                            edges.append({
                                "source": key[0], "target": key[1],
                                "weight": 0.7, "reason": f"Shared author: {author[:25]}",
                                "type": "author"
                            })

        # Build clusters from dominant MeSH terms
        clusters = {}
        for term, indices in sorted(mesh_index.items(), key=lambda x: -len(x[1])):
            if len(indices) >= 2 and len(clusters) < 6:
                clusters[term[:30]] = indices

        # Apply simple force layout (circular with jitter)
        import math, random
        random.seed(42)
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            r = 2.5 + random.uniform(-0.3, 0.3)
            node["x"] = round(r * math.cos(angle), 3)
            node["y"] = round(r * math.sin(angle), 3)

        stats = {
            "n_articles": len(nodes),
            "n_edges": len(edges),
            "n_clusters": len(clusters),
            "mesh_terms_total": len(mesh_index),
            "shared_authors": len(author_index),
        }

        return GraphData(nodes=nodes, edges=edges, clusters=clusters, stats=stats)
