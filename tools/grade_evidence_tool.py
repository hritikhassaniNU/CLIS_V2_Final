"""
CLIS-RL: GRADE Evidence Grading Tool
=====================================
Course:  INFO 7375 — Generative AI & Prompt Engineering
Student: Hritik Ram | Northeastern University
Assignment: Take-Home Final — Custom Tool Development

GRADE Methodology Reference:
    Guyatt et al. (2008). GRADE: an emerging consensus on rating
    quality of evidence and strength of recommendations.
    BMJ, 336(7650), 924-926.

This tool implements the GRADE (Grading of Recommendations,
Assessment, Development and Evaluations) framework for
systematic evidence quality assessment in clinical literature.
"""

import os
import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ── GRADE Constants ───────────────────────────────────────────
GRADE_LEVELS = {
    'HIGH':     {'score': 1.00, 'label': 'High',     'color': '#1D9E75', 'symbol': '⊕⊕⊕⊕'},
    'MODERATE': {'score': 0.75, 'label': 'Moderate', 'color': '#EF9F27', 'symbol': '⊕⊕⊕⊝'},
    'LOW':      {'score': 0.45, 'label': 'Low',       'color': '#D85A30', 'symbol': '⊕⊕⊝⊝'},
    'VERY_LOW': {'score': 0.15, 'label': 'Very Low', 'color': '#993C1D', 'symbol': '⊕⊝⊝⊝'},
}

STUDY_DESIGN_HIERARCHY = {
    'systematic_review':  {'base': 'HIGH',    'score_bonus': 0.10, 'label': 'Systematic review / meta-analysis'},
    'rct':                {'base': 'HIGH',    'score_bonus': 0.00, 'label': 'Randomized controlled trial'},
    'cohort':             {'base': 'LOW',     'score_bonus': 0.05, 'label': 'Cohort study'},
    'case_control':       {'base': 'LOW',     'score_bonus': 0.00, 'label': 'Case-control study'},
    'cross_sectional':    {'base': 'LOW',     'score_bonus': -0.05,'label': 'Cross-sectional study'},
    'case_series':        {'base': 'VERY_LOW','score_bonus': 0.05, 'label': 'Case series'},
    'case_report':        {'base': 'VERY_LOW','score_bonus': 0.00, 'label': 'Case report'},
    'expert_opinion':     {'base': 'VERY_LOW','score_bonus': -0.05,'label': 'Expert opinion / editorial'},
    'unknown':            {'base': 'VERY_LOW','score_bonus': 0.00, 'label': 'Unknown / unclassified'},
}

DOWNGRADE_FACTORS = {
    'risk_of_bias':      'Serious risk of bias detected',
    'inconsistency':     'Inconsistent results across studies',
    'indirectness':      'Indirect evidence (different population/intervention)',
    'imprecision':       'Wide confidence intervals / small sample size',
    'publication_bias':  'Likely publication bias',
}

UPGRADE_FACTORS = {
    'large_effect':      'Large magnitude of effect (OR>2 or <0.5)',
    'dose_response':     'Dose-response gradient present',
    'confounders':       'Plausible confounders would reduce effect',
}


# ── Data Classes ──────────────────────────────────────────────
@dataclass
class GradeAssessment:
    """
    Structured GRADE assessment output for a single clinical article.

    Fields follow the official GRADE framework domains:
    Population, Intervention, Comparison, Outcome (PICO).
    """
    # Article identity
    title:            str = ''
    abstract:         str = ''
    clinical_question:str = ''

    # PICO elements
    population:       str = 'Not specified'
    intervention:     str = 'Not specified'
    comparison:       str = 'Not specified'
    outcome:          str = 'Not specified'

    # Study design
    study_design:     str = 'unknown'
    study_design_label:str = ''
    sample_size:      str = 'Not reported'
    follow_up:        str = 'Not reported'

    # GRADE quality domains
    risk_of_bias:     str = 'Not assessed'
    inconsistency:    str = 'Not assessed'
    indirectness:     str = 'Not assessed'
    imprecision:      str = 'Not assessed'
    publication_bias: str = 'Not assessed'

    # Upgrade factors
    large_effect:     bool = False
    dose_response:    bool = False
    confounders:      bool = False

    # Final grade
    grade_level:      str = 'VERY_LOW'
    grade_score:      float = 0.15
    grade_label:      str = 'Very Low'
    grade_symbol:     str = '⊕⊝⊝⊝'

    # Narrative outputs
    main_finding:     str = ''
    clinical_implication: str = ''
    limitations:      str = ''
    recommendation_strength: str = 'Weak'

    # Metadata
    graded_by:        str = 'GRADE-Tool v1.0 + Groq Llama-3.3-70B'
    grading_time_ms:  float = 0.0
    confidence:       float = 0.0

    def to_dict(self):
        return asdict(self)

    def to_clinical_report(self) -> str:
        """Format as a structured clinical evidence report."""
        lines = [
            '=' * 60,
            'GRADE EVIDENCE ASSESSMENT REPORT',
            '=' * 60,
            f'Clinical question: {self.clinical_question}',
            f'Article: {self.title[:80]}{"..." if len(self.title)>80 else ""}',
            '',
            '--- PICO FRAMEWORK ---',
            f'  Population:    {self.population}',
            f'  Intervention:  {self.intervention}',
            f'  Comparison:    {self.comparison}',
            f'  Outcome:       {self.outcome}',
            '',
            '--- STUDY CHARACTERISTICS ---',
            f'  Design:        {self.study_design_label}',
            f'  Sample size:   {self.sample_size}',
            f'  Follow-up:     {self.follow_up}',
            '',
            '--- GRADE QUALITY DOMAINS ---',
            f'  Risk of bias:      {self.risk_of_bias}',
            f'  Inconsistency:     {self.inconsistency}',
            f'  Indirectness:      {self.indirectness}',
            f'  Imprecision:       {self.imprecision}',
            f'  Publication bias:  {self.publication_bias}',
            '',
            '--- EVIDENCE GRADE ---',
            f'  Level:    {self.grade_symbol}  {self.grade_label.upper()}',
            f'  Score:    {self.grade_score:.2f} / 1.00',
            f'  Strength: {self.recommendation_strength} recommendation',
            '',
            '--- CLINICAL SUMMARY ---',
            f'  Main finding:         {self.main_finding}',
            f'  Clinical implication: {self.clinical_implication}',
            f'  Key limitations:      {self.limitations}',
            '',
            f'Graded by: {self.graded_by}',
            f'Grading time: {self.grading_time_ms:.0f}ms',
            '=' * 60,
        ]
        return '\n'.join(lines)


# ── Main Tool Class ───────────────────────────────────────────
class GradeEvidenceTool:
    """
    GRADE Methodology Evidence Grading Tool for CLIS-RL.

    Integrates with Groq LLM (Llama 3.3 70B) to perform structured
    GRADE assessments of clinical literature. Produces standardized
    GradeAssessment objects for use as RL reward signals and
    clinician-facing evidence summaries.

    Usage:
        tool = GradeEvidenceTool()
        assessment = tool.grade(abstract, clinical_question)
        print(assessment.to_clinical_report())
        reward = assessment.grade_score  # use as RL reward signal

    Integration with CLIS-RL:
        Replaces the simple score_evidence_with_groq() function
        from NB3 with a full GRADE-compliant assessment that
        provides richer reward signals for both the UCB bandit
        and REINFORCE policy gradient agent.
    """

    def __init__(self, groq_api_key: Optional[str] = None, model: str = 'llama-3.3-70b-versatile'):
        self.api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.model   = model
        self.client  = None
        self._assessment_cache = {}
        self._call_count = 0

        # Try to initialize Groq client
        if self.api_key and self.api_key != 'your_groq_api_key_here':
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                print(f'GradeEvidenceTool initialized with Groq ({model})')
            except ImportError:
                print('Groq not installed — using rule-based fallback grader')
        else:
            print('GradeEvidenceTool initialized in rule-based mode (no API key)')

    # ── Public API ────────────────────────────────────────────

    def grade(self, abstract: str, clinical_question: str,
              title: str = '') -> GradeAssessment:
        """
        Main entry point. Grade a single article abstract.

        Args:
            abstract:          Full article abstract text
            clinical_question: The clinical question being answered
            title:             Article title (optional, improves accuracy)

        Returns:
            GradeAssessment dataclass with full GRADE evaluation
        """
        start = time.time()

        # Check cache (avoid re-grading identical abstracts)
        cache_key = hash(abstract[:200] + clinical_question[:100])
        if cache_key in self._assessment_cache:
            return self._assessment_cache[cache_key]

        # Detect study design from text (rule-based, fast)
        study_design = self._detect_study_design(abstract)

        # Run full GRADE assessment
        if self.client:
            assessment = self._grade_with_llm(
                abstract, clinical_question, title, study_design
            )
        else:
            assessment = self._grade_rule_based(
                abstract, clinical_question, title, study_design
            )

        # Finalize
        assessment.grading_time_ms = (time.time() - start) * 1000
        assessment.study_design_label = STUDY_DESIGN_HIERARCHY.get(
            study_design, STUDY_DESIGN_HIERARCHY['unknown']
        )['label']

        self._assessment_cache[cache_key] = assessment
        self._call_count += 1
        return assessment

    def grade_batch(self, articles: list, clinical_question: str,
                    verbose: bool = True) -> list:
        """
        Grade a list of articles and return sorted by grade score.

        Args:
            articles: List of dicts with 'abstract' and optionally 'title'
            clinical_question: Shared clinical question for all articles

        Returns:
            List of GradeAssessment objects sorted by grade_score descending
        """
        assessments = []
        for i, art in enumerate(articles):
            if verbose:
                print(f'  Grading article {i+1}/{len(articles)}...', end=' ')
            a = self.grade(
                abstract=art.get('abstract', art.get('title', '')),
                clinical_question=clinical_question,
                title=art.get('title', '')
            )
            if verbose:
                print(f'{a.grade_symbol} {a.grade_label}')
            assessments.append(a)

        # Sort by grade score descending
        assessments.sort(key=lambda x: x.grade_score, reverse=True)
        return assessments

    def compute_rl_reward(self, assessment: GradeAssessment,
                          relevance_score: float = 0.7,
                          recency_score: float = 0.7) -> float:
        """
        Convert GRADE assessment to RL reward signal.

        Combines evidence grade with relevance and recency
        using the same weighting as NB2's reward function.
        This is the integration point with the REINFORCE agent.

        Returns: float in [0, 1]
        """
        return (
            0.60 * assessment.grade_score +
            0.25 * relevance_score +
            0.15 * recency_score
        )

    def summarize_batch(self, assessments: list) -> dict:
        """
        Produce a batch summary across multiple assessments.
        Returns aggregate statistics for reporting.
        """
        if not assessments:
            return {}
        scores  = [a.grade_score for a in assessments]
        grades  = [a.grade_level for a in assessments]
        return {
            'n_articles':    len(assessments),
            'mean_score':    round(sum(scores)/len(scores), 4),
            'max_score':     max(scores),
            'min_score':     min(scores),
            'grade_counts':  {g: grades.count(g) for g in set(grades)},
            'top_grade':     assessments[0].grade_level,
            'top_article':   assessments[0].title[:60] if assessments else '',
            'pct_high_mod':  round(sum(1 for g in grades
                                       if g in ['HIGH','MODERATE'])/len(grades)*100, 1),
        }

    # ── Private: Study design detection ──────────────────────

    def _detect_study_design(self, text: str) -> str:
        """
        Rule-based study design detector.
        Faster and cheaper than LLM for this classification.
        """
        t = text.lower()

        # Highest evidence first
        if any(p in t for p in ['systematic review','meta-analysis','cochrane']):
            return 'systematic_review'
        if any(p in t for p in ['randomized','randomised','rct','random assignment',
                                  'placebo-controlled','double-blind','double blind']):
            return 'rct'
        if any(p in t for p in ['prospective cohort','retrospective cohort',
                                  'cohort study','follow-up study','longitudinal']):
            return 'cohort'
        if any(p in t for p in ['case-control','case control','odds ratio']):
            return 'case_control'
        if any(p in t for p in ['cross-sectional','cross sectional','prevalence study']):
            return 'cross_sectional'
        if any(p in t for p in ['case series','consecutive patients','n=']):
            return 'case_series'
        if any(p in t for p in ['case report','we report a','we describe a']):
            return 'case_report'
        if any(p in t for p in ['expert opinion','editorial','commentary','review article']):
            return 'expert_opinion'
        return 'unknown'

    # ── Private: LLM-based grading ────────────────────────────

    def _grade_with_llm(self, abstract: str, question: str,
                         title: str, study_design: str) -> GradeAssessment:
        """Full GRADE assessment using Groq LLM."""

        prompt = f"""You are a clinical evidence grader using the GRADE methodology.
Assess the following article abstract and return ONLY a valid JSON object.

Clinical question: {question}
Article title: {title}
Abstract: {abstract}
Detected study design: {study_design}

Return EXACTLY this JSON structure (no markdown, no explanation):
{{
  "population": "Who was studied (age, condition, setting)",
  "intervention": "What intervention/exposure was studied",
  "comparison": "What was it compared to (or 'No comparator')",
  "outcome": "Primary outcome measured",
  "sample_size": "Number of participants (e.g. n=500) or 'Not reported'",
  "follow_up": "Duration of follow-up or 'Not reported'",
  "risk_of_bias": "Not serious / Serious / Very serious — explain in 1 sentence",
  "inconsistency": "Not serious / Serious — explain in 1 sentence",
  "indirectness": "Not serious / Serious — explain in 1 sentence",
  "imprecision": "Not serious / Serious — explain in 1 sentence",
  "publication_bias": "Undetected / Suspected — explain in 1 sentence",
  "large_effect": true or false,
  "dose_response": true or false,
  "confounders": true or false,
  "main_finding": "1-sentence main result",
  "clinical_implication": "1-sentence clinical takeaway",
  "limitations": "1-sentence key limitation",
  "confidence": 0.0 to 1.0
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=600
            )
            raw  = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw  = re.sub(r'```json|```', '', raw).strip()
            data = json.loads(raw)
            return self._build_assessment(data, abstract, question, title, study_design)

        except Exception as e:
            print(f'  LLM grading error: {e} — falling back to rule-based')
            return self._grade_rule_based(abstract, question, title, study_design)

    # ── Private: Rule-based fallback grading ─────────────────

    def _grade_rule_based(self, abstract: str, question: str,
                           title: str, study_design: str) -> GradeAssessment:
        """
        Rule-based GRADE assessment — no API key needed.
        Used as fallback when Groq unavailable.
        Less nuanced but fully functional for RL reward signal.
        """
        t = abstract.lower()

        # Detect downgrade factors from text
        risk_of_bias = ('Serious — open-label or no blinding mentioned'
                        if any(p in t for p in ['open-label','unblinded','no control'])
                        else 'Not serious — controlled design inferred')

        inconsistency = ('Serious — heterogeneity or conflicting results mentioned'
                         if any(p in t for p in ['heterogeneity','conflicting','inconsistent'])
                         else 'Not serious — consistent findings')

        indirectness = ('Serious — surrogate endpoint or indirect population'
                        if any(p in t for p in ['surrogate','proxy','indirect'])
                        else 'Not serious — direct evidence')

        imprecision = ('Serious — small sample or wide CI'
                       if any(p in t for p in ['n=1','n=2','n=3','n=4','n=5',
                                                'wide confidence','small sample'])
                       else 'Not serious — adequate sample size')

        pub_bias = ('Suspected — single positive trial'
                    if study_design == 'case_report'
                    else 'Undetected')

        # Upgrade factors
        large_effect  = any(p in t for p in ['large effect','substantial','marked reduction',
                                               'significantly reduced','or >2','or <0.5'])
        dose_response = any(p in t for p in ['dose-response','dose response','gradient'])
        confounders   = any(p in t for p in ['adjusted for','controlling for','multivariable'])

        # Extract sample size
        n_match = re.search(r'n\s*=\s*(\d+)', t) or re.search(r'(\d+)\s+patients', t)
        sample_size = f'n={n_match.group(1)}' if n_match else 'Not reported'

        data = {
            'population':         'Inferred from study context',
            'intervention':       'See abstract',
            'comparison':         'See abstract',
            'outcome':            'See abstract',
            'sample_size':        sample_size,
            'follow_up':          'Not extracted',
            'risk_of_bias':       risk_of_bias,
            'inconsistency':      inconsistency,
            'indirectness':       indirectness,
            'imprecision':        imprecision,
            'publication_bias':   pub_bias,
            'large_effect':       large_effect,
            'dose_response':      dose_response,
            'confounders':        confounders,
            'main_finding':       abstract[:120] + '...' if len(abstract) > 120 else abstract,
            'clinical_implication': 'Review full article for clinical guidance.',
            'limitations':        'Rule-based assessment — LLM grading recommended.',
            'confidence':         0.65,
        }
        return self._build_assessment(data, abstract, question, title, study_design)

    # ── Private: Build assessment from parsed data ────────────

    def _build_assessment(self, data: dict, abstract: str, question: str,
                           title: str, study_design: str) -> GradeAssessment:
        """
        Convert parsed LLM/rule output into a GradeAssessment.
        Applies GRADE upgrade/downgrade logic to compute final grade.
        """
        # Start from base grade for study design
        design_info  = STUDY_DESIGN_HIERARCHY.get(study_design,
                                                    STUDY_DESIGN_HIERARCHY['unknown'])
        base_grade   = design_info['base']
        score        = GRADE_LEVELS[base_grade]['score'] + design_info['score_bonus']

        # Apply downgrade factors (-0.15 each, max 2 levels down)
        downgrades = 0
        for factor in ['risk_of_bias','inconsistency','indirectness',
                        'imprecision','publication_bias']:
            val = str(data.get(factor, '')).lower()
            if 'serious' in val or 'suspected' in val:
                score     -= 0.12
                downgrades += 1

        # Apply upgrade factors (+0.10 each, max 1 level up)
        upgrades = 0
        for factor in ['large_effect','dose_response','confounders']:
            if data.get(factor, False):
                score    += 0.08
                upgrades += 1

        # Clamp score and determine final grade level
        score = round(float(max(0.05, min(1.0, score))), 4)
        if score >= 0.90:   grade_level = 'HIGH'
        elif score >= 0.65: grade_level = 'MODERATE'
        elif score >= 0.35: grade_level = 'LOW'
        else:               grade_level = 'VERY_LOW'

        grade_info = GRADE_LEVELS[grade_level]

        # Recommendation strength (GRADE strong vs weak)
        rec_strength = ('Strong' if grade_level in ['HIGH','MODERATE']
                        else 'Weak')

        return GradeAssessment(
            title=title,
            abstract=abstract[:500],
            clinical_question=question,
            population=str(data.get('population','Not extracted')),
            intervention=str(data.get('intervention','Not extracted')),
            comparison=str(data.get('comparison','Not extracted')),
            outcome=str(data.get('outcome','Not extracted')),
            study_design=study_design,
            sample_size=str(data.get('sample_size','Not reported')),
            follow_up=str(data.get('follow_up','Not reported')),
            risk_of_bias=str(data.get('risk_of_bias','Not assessed')),
            inconsistency=str(data.get('inconsistency','Not assessed')),
            indirectness=str(data.get('indirectness','Not assessed')),
            imprecision=str(data.get('imprecision','Not assessed')),
            publication_bias=str(data.get('publication_bias','Not assessed')),
            large_effect=bool(data.get('large_effect', False)),
            dose_response=bool(data.get('dose_response', False)),
            confounders=bool(data.get('confounders', False)),
            grade_level=grade_level,
            grade_score=score,
            grade_label=grade_info['label'],
            grade_symbol=grade_info['symbol'],
            main_finding=str(data.get('main_finding','')),
            clinical_implication=str(data.get('clinical_implication','')),
            limitations=str(data.get('limitations','')),
            recommendation_strength=rec_strength,
            confidence=float(data.get('confidence', 0.7)),
            graded_by=f'GRADE-Tool v1.0 + Groq {self.model}'
                      if self.client else 'GRADE-Tool v1.0 (rule-based)',
        )
