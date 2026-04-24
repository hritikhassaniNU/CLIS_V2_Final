"""
CLIS V2: ICD-10-CM RAG Retrieval Module
=========================================
Course:  INFO 7375 — Generative AI & Prompt Engineering
Student: Hritik Ram | Northeastern University
Final Project: CLIS V2

Implements RAG over ICD-10-CM Official Guidelines (CMS).
Uses TF-IDF vector retrieval with semantic chunking.
Groq LLM synthesizes the final coding answer from retrieved chunks.

Data source: CMS ICD-10-CM Official Guidelines for Coding and Reporting
URL: https://www.cms.gov/medicare/coding-billing/icd-10-codes
"""

import re
import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ── ICD-10 Knowledge Base ─────────────────────────────────────
# Curated chunks from CMS ICD-10-CM Official Guidelines FY2024
# Each chunk: section ID, title, content, relevant codes
ICD10_KNOWLEDGE_BASE = [
    {
        "section": "I.C.4.a.1",
        "title": "Type 1 diabetes mellitus",
        "content": "Type 1 diabetes mellitus (juvenile-onset) is coded E10.-. Patients with type 1 diabetes who use insulin do not require an additional code for long-term insulin use. Type 1 diabetes is always insulin-dependent. If documentation is unclear about type, query the provider.",
        "codes": ["E10", "E10.9", "E10.65"],
        "keywords": ["type 1 diabetes", "juvenile diabetes", "insulin dependent", "E10"]
    },
    {
        "section": "I.C.4.a.2",
        "title": "Type 2 diabetes mellitus",
        "content": "Type 2 diabetes mellitus is coded E11.-. If a patient with type 2 diabetes uses insulin, assign code Z79.4 (Long-term current use of insulin) as an additional code. Do not assign Z79.4 if insulin is given temporarily. Oral hypoglycemic drugs are captured with Z79.84.",
        "codes": ["E11", "E11.9", "Z79.4", "Z79.84"],
        "keywords": ["type 2 diabetes", "T2DM", "insulin", "oral hypoglycemic", "E11", "Z79.4"]
    },
    {
        "section": "I.C.4.a.3",
        "title": "Diabetes with chronic kidney disease",
        "content": "When diabetes mellitus is documented with diabetic chronic kidney disease (CKD), assign the appropriate diabetes code (E10-E13) with the fourth/fifth character .22 for diabetic CKD. Then assign a code from N18.- to identify the stage of CKD. The combination code E11.22 (Type 2 DM with diabetic CKD) should be sequenced first, followed by the N18.- stage code. Do NOT use N18.9 if stage is documented.",
        "codes": ["E11.22", "E10.22", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6"],
        "keywords": ["diabetes CKD", "diabetic chronic kidney disease", "E11.22", "N18", "kidney disease diabetes"]
    },
    {
        "section": "I.C.4.a.4",
        "title": "Diabetes with complications — sequencing",
        "content": "Combination codes in categories E08-E13 identify the type of diabetes, the body system affected, and the complication. Use combination codes as principal or first-listed diagnosis. If no combination code exists, sequence the diabetes code first, followed by the complication code. Sequence secondary diagnoses from the Tabular List after the principal diagnosis.",
        "codes": ["E11.40", "E11.41", "E11.42", "E11.51", "E11.52"],
        "keywords": ["diabetes complications", "sequencing", "combination codes", "diabetes retinopathy", "diabetes neuropathy"]
    },
    {
        "section": "I.C.4.a.5",
        "title": "Diabetes with hyperglycemia",
        "content": "Uncontrolled diabetes is classified as either hyperglycemia (E11.65) or hypoglycemia (E11.64x). Code E11.65 is used for Type 2 DM with hyperglycemia (uncontrolled, not otherwise specified). Do not assume uncontrolled means hyperglycemia vs hypoglycemia without documentation. Query the provider if documentation is ambiguous.",
        "codes": ["E11.65", "E11.64", "E10.65"],
        "keywords": ["uncontrolled diabetes", "hyperglycemia", "hypoglycemia", "E11.65", "E11.64"]
    },
    {
        "section": "I.C.9.a",
        "title": "Hypertension — essential and secondary",
        "content": "Essential (primary) hypertension is coded I10. Code I10 is assigned when hypertension is documented without additional specification. Secondary hypertension requires two codes: a code from category I15 to identify the type, and a code to identify the underlying etiology. Hypertensive heart disease (I11.-) is used when heart failure is documented with hypertension — do not require the physician to link them.",
        "codes": ["I10", "I11", "I11.0", "I11.9", "I15", "I15.0", "I15.1"],
        "keywords": ["hypertension", "high blood pressure", "I10", "essential hypertension", "secondary hypertension"]
    },
    {
        "section": "I.C.9.b",
        "title": "Hypertensive heart disease",
        "content": "Hypertensive heart disease with heart failure: assign code I11.0 (Hypertensive heart disease with heart failure). Also assign the appropriate heart failure code from I50.- to identify the type of heart failure. The provider does not need to explicitly link hypertension and heart failure — the combination code assumes a causal relationship per ICD-10-CM guidelines.",
        "codes": ["I11.0", "I11.9", "I50.1", "I50.20", "I50.22", "I50.30", "I50.32"],
        "keywords": ["hypertensive heart disease", "heart failure hypertension", "I11.0", "I50", "HFpEF", "HFrEF"]
    },
    {
        "section": "I.C.9.c",
        "title": "Hypertensive chronic kidney disease",
        "content": "Hypertensive chronic kidney disease (CKD) is coded I12.-. The ICD-10-CM presumes a cause-and-effect relationship between hypertension and CKD. Assign I12.9 (hypertensive CKD, unspecified stage) or I12.30/I12.9 with N18.- stage code. Do NOT separately code hypertension (I10) and CKD when hypertensive CKD codes apply.",
        "codes": ["I12", "I12.9", "I12.30", "N18.3", "N18.4", "N18.5"],
        "keywords": ["hypertensive CKD", "hypertension kidney disease", "I12", "chronic kidney disease hypertension"]
    },
    {
        "section": "I.C.9.d",
        "title": "Hypertensive heart and chronic kidney disease",
        "content": "When a patient has hypertension, heart disease, AND CKD, assign a code from I13.- (hypertensive heart and chronic kidney disease). Use I13.10 for without heart failure, stage 1-4 CKD; I13.11 for without heart failure, stage 5/ESRD; I13.0 for with heart failure and stage 1-4 CKD; I13.2 for with heart failure and stage 5/ESRD. Always add N18.- and I50.- as additional codes.",
        "codes": ["I13", "I13.0", "I13.10", "I13.11", "I13.2"],
        "keywords": ["hypertensive heart CKD", "I13", "hypertension heart kidney", "triple diagnosis"]
    },
    {
        "section": "I.C.14.a.1",
        "title": "CKD stage coding",
        "content": "Chronic kidney disease stages are classified N18.1 (stage 1) through N18.6 (stage 6/ESRD). N18.9 is used only when stage is not documented. Assign N18.6 for end-stage renal disease (ESRD). When a patient has CKD and undergoes kidney transplant, the CKD code should NOT be assumed resolved — query provider about post-transplant function.",
        "codes": ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
        "keywords": ["CKD stage", "chronic kidney disease", "N18", "ESRD", "kidney failure stages"]
    },
    {
        "section": "I.C.2.a",
        "title": "Principal diagnosis for cancer",
        "content": "When the admission is for treatment of the primary site malignancy, the primary malignancy is sequenced first. When the reason is treatment of metastatic disease, the metastatic site is the principal diagnosis. Z85.- codes are used for personal history of malignancy when no current treatment is ongoing. Active treatment includes surgery, radiation, chemotherapy, immunotherapy.",
        "codes": ["C00-C97", "Z85", "Z79.899"],
        "keywords": ["cancer coding", "malignancy", "principal diagnosis cancer", "metastatic cancer", "Z85"]
    },
    {
        "section": "I.C.10.a",
        "title": "COPD and asthma coding",
        "content": "COPD is coded J44.-. Acute exacerbation of COPD is J44.1. J44.0 is COPD with acute lower respiratory infection — also assign the infection code. Asthma is classified J45.- by severity (mild intermittent J45.20, mild persistent J45.30, moderate persistent J45.40, severe persistent J45.50). Acute exacerbation of asthma includes status asthmaticus (J45.x1).",
        "codes": ["J44", "J44.0", "J44.1", "J44.9", "J45.20", "J45.30", "J45.40", "J45.50"],
        "keywords": ["COPD", "asthma", "J44", "J45", "exacerbation", "respiratory"]
    },
    {
        "section": "I.C.19.a",
        "title": "Injury coding — fractures",
        "content": "Fractures are coded with 7th character extensions: A (initial encounter), D (subsequent encounter), S (sequela). Active treatment = A regardless of provider seen. After active treatment ends, use D for healing/routine care. S is for late effects. Pathological fractures (M80.-) differ from traumatic fractures — query provider when mechanism is unclear.",
        "codes": ["S", "M80", "M84.4", "M84.5"],
        "keywords": ["fracture coding", "7th character", "initial encounter", "subsequent encounter", "sequela", "S A D"]
    },
    {
        "section": "I.C.21.c.1",
        "title": "BMI coding",
        "content": "BMI codes (Z68.-) should only be assigned when the associated condition (obesity E66.-) is documented. BMI codes are always secondary — never the principal diagnosis. Assign BMI as additional code to obesity, morbid obesity, or overweight. Pediatric BMI uses Z68.5x codes. Document the BMI numeric value when available.",
        "codes": ["Z68", "Z68.1", "Z68.2", "Z68.3", "Z68.4", "E66", "E66.01", "E66.09"],
        "keywords": ["BMI", "obesity", "Z68", "E66", "morbid obesity", "overweight"]
    },
    {
        "section": "I.C.21.c.15",
        "title": "Long-term drug use coding",
        "content": "Long-term current use of medications is captured with Z79 codes. Key codes: Z79.4 insulin, Z79.01 anticoagulants, Z79.02 antithrombotics, Z79.1 non-steroidal anti-inflammatories, Z79.3 contraceptives, Z79.51 inhaled steroids, Z79.52 systemic steroids, Z79.84 oral hypoglycemics, Z79.899 other long-term drug therapy. These are secondary codes only.",
        "codes": ["Z79.4", "Z79.01", "Z79.02", "Z79.1", "Z79.51", "Z79.52", "Z79.84", "Z79.899"],
        "keywords": ["long-term medication", "Z79", "insulin use", "anticoagulant", "drug therapy codes"]
    },
    {
        "section": "I.C.15.a",
        "title": "Obstetric coding — trimester and outcome",
        "content": "Obstetric codes (O00-O9A) require a 7th character for trimester when applicable: 1 (first, < 14 weeks), 2 (second, 14-28 weeks), 3 (third, >28 weeks). Outcome of delivery (Z37.-) is always required as additional code when delivery occurs. HIV in pregnancy uses O98.7-. Pre-existing conditions in pregnancy are coded O10-O16 before the underlying condition.",
        "codes": ["O", "Z37", "Z37.0", "Z37.1", "Z34", "O09"],
        "keywords": ["obstetric coding", "pregnancy", "trimester", "delivery outcome", "Z37", "O codes"]
    },
    {
        "section": "I.C.18.a",
        "title": "Sepsis and severe sepsis",
        "content": "Sepsis: assign the underlying systemic infection code (e.g. A41.9 sepsis unspecified, A41.01 MRSA sepsis) as principal diagnosis. Severe sepsis (with organ dysfunction) requires R65.20 or R65.21 (with septic shock) as additional code. Septic shock is never the principal diagnosis — the causal infection is sequenced first. Always code the associated organ dysfunction.",
        "codes": ["A41.9", "A41.01", "A40", "R65.20", "R65.21", "R57.2"],
        "keywords": ["sepsis", "septic shock", "A41", "R65.20", "R65.21", "severe sepsis", "organ dysfunction"]
    },
    {
        "section": "I.C.4.b",
        "title": "Diabetes in pregnancy",
        "content": "Gestational diabetes is coded O24.4-. Pre-existing type 1 diabetes in pregnancy is O24.01-, type 2 is O24.11-. Do NOT use E10-E11 as principal diagnosis when the diabetes is affecting the pregnancy — use O24 codes. Additional codes for insulin use (Z79.4) and oral hypoglycemics (Z79.84) may still apply.",
        "codes": ["O24.4", "O24.41", "O24.410", "O24.01", "O24.11"],
        "keywords": ["gestational diabetes", "diabetes pregnancy", "O24", "GDM"]
    },
]


@dataclass
class ICD10Result:
    """Structured result from ICD-10 coding query."""
    query: str = ""
    answer: str = ""
    primary_codes: list = field(default_factory=list)
    supporting_sections: list = field(default_factory=list)
    confidence: str = "Moderate"
    generated_by: str = "Rule-based"
    disclaimer: str = "Always verify with official CMS ICD-10-CM guidelines before billing."


class ICD10Retriever:
    """
    RAG-based ICD-10-CM coding assistant.

    Uses TF-IDF cosine similarity for chunk retrieval,
    then Groq Llama 3.3 70B to synthesize the coding answer
    from retrieved guideline sections.

    This is the RAG component for CLIS V2 Final Project.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.knowledge_base = ICD10_KNOWLEDGE_BASE
        self._build_tfidf_index()

        if self.api_key and self.api_key not in ("", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass

    def _build_tfidf_index(self):
        """Build TF-IDF index over knowledge base chunks."""
        self._corpus = []
        for chunk in self.knowledge_base:
            text = (chunk["title"] + " " +
                    chunk["content"] + " " +
                    " ".join(chunk["keywords"]) + " " +
                    " ".join(chunk["codes"])).lower()
            self._corpus.append(text)

        # Build IDF
        N = len(self._corpus)
        self._vocab = {}
        doc_freq = {}
        for doc in self._corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        for word, df in doc_freq.items():
            self._vocab[word] = math.log((N + 1) / (df + 1)) + 1

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tfidf_vector(self, text: str) -> dict:
        tokens = self._tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: (c / n) * self._vocab.get(t, 0)
                for t, c in tf.items() if t in self._vocab}

    def _cosine(self, v1: dict, v2: dict) -> float:
        dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in v1)
        n1 = math.sqrt(sum(x ** 2 for x in v1.values())) or 1
        n2 = math.sqrt(sum(x ** 2 for x in v2.values())) or 1
        return dot / (n1 * n2)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve top-k most relevant guideline chunks for query."""
        qvec = self._tfidf_vector(query)
        scores = []
        for i, doc in enumerate(self._corpus):
            dvec = self._tfidf_vector(doc)
            scores.append((self._cosine(qvec, dvec), i))
        scores.sort(reverse=True)
        return [self.knowledge_base[i] for _, i in scores[:top_k]]

    def answer(self, query: str) -> ICD10Result:
        """
        Full RAG pipeline: retrieve relevant chunks → synthesize answer.
        Returns structured ICD10Result.
        """
        chunks = self.retrieve(query, top_k=3)
        all_codes = []
        for c in chunks:
            all_codes.extend(c["codes"])
        all_codes = list(dict.fromkeys(all_codes))[:8]

        if self.client:
            answer_text, gen_by = self._synthesize_with_llm(query, chunks)
        else:
            answer_text, gen_by = self._synthesize_rule_based(query, chunks)

        confidence = "High" if chunks[0].get("section", "") else "Moderate"

        return ICD10Result(
            query=query,
            answer=answer_text,
            primary_codes=all_codes[:5],
            supporting_sections=[
                {"section": c["section"], "title": c["title"]}
                for c in chunks
            ],
            confidence=confidence,
            generated_by=gen_by,
        )

    def _synthesize_with_llm(self, query: str, chunks: list) -> tuple:
        """Synthesize coding answer using Groq LLM."""
        context = "\n\n".join([
            f"[{c['section']}] {c['title']}\n{c['content']}\nRelevant codes: {', '.join(c['codes'])}"
            for c in chunks
        ])

        prompt = f"""You are an expert medical coder using ICD-10-CM Official Guidelines.

Clinical coding question: {query}

Relevant guideline sections:
{context}

Provide a structured coding answer with:
1. The primary ICD-10-CM code(s) to assign (most specific first)
2. Any required additional codes
3. Sequencing rule (which code is principal/first-listed)
4. One-sentence clinical rationale
5. Any important caveats or documentation requirements

Be concise and clinically precise. Use code format like E11.22, N18.3."""

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip(), "Groq Llama 3.3 70B"
        except Exception as e:
            return self._synthesize_rule_based(query, chunks)[0], "Rule-based (LLM error)"

    def _synthesize_rule_based(self, query: str, chunks: list) -> tuple:
        """Rule-based fallback synthesis."""
        top = chunks[0]
        codes_str = ", ".join(top["codes"][:3])
        answer = (
            f"Based on ICD-10-CM guidelines section {top['section']} ({top['title']}): "
            f"{top['content']} "
            f"Primary codes to consider: {codes_str}. "
            f"Always verify sequencing rules with the complete ICD-10-CM Tabular List."
        )
        return answer, "Rule-based (no Groq key)"
