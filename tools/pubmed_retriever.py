"""
CLIS-RL: PubMed NCBI API Module
=================================
Course:  INFO 7375 — Generative AI & Prompt Engineering
Student: Hritik Ram | Northeastern University

Real PubMed retrieval via NCBI E-utilities API.
No API key required for basic queries (up to 3 requests/second).
With NCBI API key: up to 10 requests/second.

NCBI E-utilities documentation:
    https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import re
import os
from dataclasses import dataclass, field
from typing import Optional

# ── NCBI API Config ───────────────────────────────────────────
NCBI_BASE_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY   = os.getenv("NCBI_API_KEY", "")   # optional — works without it
DEFAULT_DELAY  = 0.4   # seconds between requests (respects NCBI rate limit)
MAX_RETRIES    = 3


@dataclass
class PubMedArticle:
    """
    Structured representation of a PubMed article.
    Compatible with CLIS-RL pipeline and GradeEvidenceTool.
    """
    pmid:         str   = ''
    title:        str   = ''
    abstract:     str   = ''
    authors:      list  = field(default_factory=list)
    journal:      str   = ''
    pub_year:     str   = ''
    pub_date:     str   = ''
    mesh_terms:   list  = field(default_factory=list)
    pub_types:    list  = field(default_factory=list)
    doi:          str   = ''
    url:          str   = ''

    # Computed fields
    recency_score:    float = 0.5
    relevance_score:  float = 0.5
    grade_val:        float = 0.5
    grade:            str   = 'C'

    def to_dict(self):
        return {
            'pmid':          self.pmid,
            'title':         self.title,
            'abstract':      self.abstract[:300] + '...' if len(self.abstract) > 300
                             else self.abstract,
            'authors':       self.authors[:3],
            'journal':       self.journal,
            'pub_year':      self.pub_year,
            'grade':         self.grade,
            'grade_val':     self.grade_val,
            'recency_score': self.recency_score,
            'url':           self.url,
        }


class PubMedRetriever:
    """
    Real PubMed retrieval via NCBI E-utilities API.

    Implements the same interface as the simulated retrieval
    in NB3 so it can be dropped in as a direct replacement.

    Usage:
        retriever = PubMedRetriever()
        articles  = retriever.search("metformin type 2 diabetes", max_results=5)
        for art in articles:
            print(art.title, art.grade)

    Query strategy integration:
        The UCB bandit selects a query strategy arm.
        build_query() converts arm_id + clinical question
        into an optimized PubMed query string.
    """

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay       = delay
        self.last_call   = 0.0
        self.total_calls = 0
        self.cache       = {}

    # ── Public API ────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5,
               arm_id: Optional[int] = None) -> list:
        """
        Search PubMed and return structured article list.

        Args:
            query:       Clinical search query string
            max_results: Max articles to retrieve (default 5)
            arm_id:      UCB bandit arm ID (applies query strategy)

        Returns:
            List of PubMedArticle objects sorted by relevance
        """
        # Apply bandit query strategy if arm provided
        if arm_id is not None:
            query = self._apply_strategy(query, arm_id)

        # Check cache
        cache_key = f"{query}:{max_results}"
        if cache_key in self.cache:
            print(f'  [Cache hit] {query[:50]}')
            return self.cache[cache_key]

        print(f'  Querying PubMed: "{query[:60]}"')

        try:
            # Step 1: ESearch — get PMIDs
            pmids = self._esearch(query, max_results)
            if not pmids:
                print(f'  No results found for query.')
                return []

            print(f'  Found {len(pmids)} articles. Fetching details...')

            # Step 2: EFetch — get full records
            articles = self._efetch(pmids)

            # Step 3: Score and sort
            articles = self._score_articles(articles, query)
            articles.sort(key=lambda x: x.relevance_score, reverse=True)

            self.cache[cache_key] = articles
            return articles

        except Exception as e:
            print(f'  PubMed API error: {e}')
            return []

    def build_query(self, clinical_question: str, arm_id: int) -> str:
        """
        Build an optimized PubMed query from a clinical question
        using the strategy selected by the UCB bandit.

        This is the key integration between the bandit arm selection
        and real PubMed retrieval.
        """
        return self._apply_strategy(clinical_question, arm_id)

    # ── Private: Query strategy builder ──────────────────────

    def _apply_strategy(self, question: str, arm_id: int) -> str:
        """
        Apply UCB bandit arm strategy to build optimized query.

        Arm 0: MeSH + RCT filter       → best for drug efficacy
        Arm 1: Keyword + Date range     → best for epidemiology
        Arm 2: Author + Journal filter  → best for mechanisms
        Arm 3: Boolean AND + Population → best for comparisons
        Arm 4: Systematic review filter → best for treatment comparison
        """
        # Keep meaningful clinical keywords — don't strip numbers
        stopwords = {'is','are','does','what','how','why','the','a','an',
                    'in','for','of','with','and','or','vs','versus'}
        words = [w.lower().strip('?.,') for w in question.split()
                if w.lower().strip('?.,') not in stopwords and len(w) > 1]
        keywords = ' '.join(words[:5])

        strategies = {
            0: f'({keywords}) AND randomized controlled trial[pt]',
            1: f'({keywords}) AND ("2020"[dp] : "2024"[dp])',
            2: f'({keywords}) AND (mechanism[ti] OR pathway[ti] OR inhibit[ti])',
            3: f'({keywords}) AND (compare[ti] OR versus[ti] OR comparison[ti])',
            4: f'({keywords}) AND (systematic review[pt] OR meta-analysis[pt])',
        }
        return strategies.get(arm_id, keywords)
        
    # ── Private: NCBI API calls ───────────────────────────────

    def _rate_limit(self):
        """Respect NCBI rate limit — max 3 req/sec without key."""
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()
        self.total_calls += 1

    def _get(self, url: str) -> str:
        """HTTP GET with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'CLIS-RL/1.0 (educational; hritik@northeastern.edu)'}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode('utf-8')
            except urllib.error.URLError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return ''

    def _esearch(self, query: str, max_results: int) -> list:
        """ESearch: get PMIDs for a query."""
        params = {
            'db':      'pubmed',
            'term':    query,
            'retmax':  max_results,
            'retmode': 'json',
            'sort':    'relevance',
        }
        if NCBI_API_KEY:
            params['api_key'] = NCBI_API_KEY

        url  = f"{NCBI_BASE_URL}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        resp = self._get(url)
        data = json.loads(resp)
        return data.get('esearchresult', {}).get('idlist', [])

    def _efetch(self, pmids: list) -> list:
        """EFetch: get full article records for a list of PMIDs."""
        params = {
            'db':      'pubmed',
            'id':      ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract',
        }
        if NCBI_API_KEY:
            params['api_key'] = NCBI_API_KEY

        url  = f"{NCBI_BASE_URL}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        xml  = self._get(url)
        return self._parse_xml(xml, pmids)

    def _parse_xml(self, xml: str, pmids: list) -> list:
        """
        Parse PubMed XML response into PubMedArticle objects.
        Uses regex parsing — no external XML library needed.
        """
        articles = []

        # Split into individual article records
        records = re.split(r'<PubmedArticle>', xml)[1:]

        for i, record in enumerate(records):
            pmid = pmids[i] if i < len(pmids) else f'unknown_{i}'

            # Extract fields using regex
            title    = self._extract(record, r'<ArticleTitle>(.*?)</ArticleTitle>')
            abstract = self._extract_abstract(record)
            journal  = self._extract(record, r'<Title>(.*?)</Title>')
            year     = self._extract(record, r'<Year>(.*?)</Year>')
            doi      = self._extract(record, r'<ELocationID EIdType="doi".*?>(.*?)</ELocationID>')

            # Authors
            author_matches = re.findall(
                r'<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>',
                record, re.DOTALL
            )
            authors = [f"{ln} {fn}" for ln, fn in author_matches[:5]]

            # MeSH terms
            mesh = re.findall(r'<DescriptorName[^>]*>(.*?)</DescriptorName>', record)

            # Publication types
            pub_types = re.findall(r'<PublicationType[^>]*>(.*?)</PublicationType>', record)

            if title or abstract:
                articles.append(PubMedArticle(
                    pmid=pmid,
                    title=self._clean(title),
                    abstract=self._clean(abstract),
                    authors=authors,
                    journal=self._clean(journal),
                    pub_year=year,
                    pub_date=year,
                    mesh_terms=mesh[:8],
                    pub_types=pub_types[:4],
                    doi=doi,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))

        return articles

    def _extract(self, text: str, pattern: str) -> str:
        """Extract first match of a regex pattern."""
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ''

    def _extract_abstract(self, record: str) -> str:
        """Extract and concatenate abstract text sections."""
        parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>',
                           record, re.DOTALL)
        return ' '.join(self._clean(p) for p in parts)

    def _clean(self, text: str) -> str:
        """Remove XML tags and clean whitespace."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ── Private: Scoring ──────────────────────────────────────

    def _score_articles(self, articles: list, query: str) -> list:
        """
        Score articles for recency, relevance, and evidence grade.
        Produces the same fields expected by the REINFORCE agent.
        """
        current_year = 2024
        query_words  = set(query.lower().split())

        for art in articles:
            # Recency score: 0.3 (old) to 1.0 (2024)
            try:
                year = int(art.pub_year)
                art.recency_score = round(
                    min(1.0, max(0.3, (year - 2015) / (current_year - 2015))), 3
                )
            except (ValueError, ZeroDivisionError):
                art.recency_score = 0.5

            # Relevance score: keyword overlap
            title_words = set((art.title + ' ' + art.abstract[:200]).lower().split())
            overlap = len(query_words & title_words)
            art.relevance_score = round(min(1.0, overlap / max(len(query_words), 1)), 3)

            # Evidence grade from publication type
            pt_lower = [p.lower() for p in art.pub_types]
            mesh_lower = [m.lower() for m in art.mesh_terms]
            all_text = ' '.join(pt_lower + mesh_lower + [art.abstract[:100].lower()])

            if any(p in all_text for p in ['systematic review','meta-analysis']):
                art.grade, art.grade_val = 'A', 1.00
            elif any(p in all_text for p in ['randomized controlled','clinical trial']):
                art.grade, art.grade_val = 'A', 0.90
            elif any(p in all_text for p in ['cohort','longitudinal','prospective']):
                art.grade, art.grade_val = 'B', 0.70
            elif any(p in all_text for p in ['case-control','observational']):
                art.grade, art.grade_val = 'C', 0.45
            else:
                art.grade, art.grade_val = 'C', 0.40

        return articles
