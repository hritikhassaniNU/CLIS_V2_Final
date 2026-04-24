"""
CLIS V2 — RLHF Feedback Loop + Persistent Bandit
===================================================
Student: Hritik Ram | Northeastern University

Two production upgrades:
1. RLHF: Clinician thumbs-up/down updates bandit reward
   estimates in real time across sessions
2. Persistent bandit: SQLite-backed state so the policy
   improves with every query, not just within one session

This makes CLIS genuinely adaptive — the system gets
smarter the more it's used.
"""

import sqlite3, json, math, time, os
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedbackRecord:
    session_id: str
    query: str
    arm_id: int
    context_id: int
    reward: float
    user_rating: Optional[int]   # 1=thumbs up, -1=thumbs down, None=no rating
    timestamp: float
    top_grade: str


class PersistentBandit:
    """
    UCB bandit with SQLite-backed persistent state.

    The bandit policy persists across Streamlit sessions via SQLite.
    RLHF feedback (thumbs up/down) adjusts reward estimates in real time.

    Schema:
        bandit_state: per (context, arm) mean reward + count
        feedback:     per-query user ratings
        sessions:     session-level metadata
    """

    N_CONTEXTS = 4
    N_ARMS     = 5
    UCB_C      = 2.0   # exploration constant
    RLHF_LR    = 0.15  # how strongly user feedback shifts reward

    def __init__(self, db_path: str = "clis_bandit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bandit_state (
                context_id INTEGER,
                arm_id     INTEGER,
                mean_reward REAL DEFAULT 0.5,
                n_pulls     INTEGER DEFAULT 0,
                PRIMARY KEY (context_id, arm_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query      TEXT,
                arm_id     INTEGER,
                context_id INTEGER,
                reward     REAL,
                user_rating INTEGER,
                top_grade  TEXT,
                timestamp  REAL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL,
                n_queries  INTEGER DEFAULT 0
            );
        """)
        # Seed initial state if empty
        cur = conn.execute("SELECT COUNT(*) FROM bandit_state")
        if cur.fetchone()[0] == 0:
            for ctx in range(self.N_CONTEXTS):
                for arm in range(self.N_ARMS):
                    # Domain-informed priors
                    prior = self._domain_prior(ctx, arm)
                    conn.execute(
                        "INSERT INTO bandit_state VALUES(?,?,?,?)",
                        (ctx, arm, prior, 1)
                    )
        conn.commit()
        conn.close()

    def _domain_prior(self, ctx: int, arm: int) -> float:
        """
        Domain-informed initial reward estimates.
        Encodes clinical knowledge about which query strategies
        work best for each context type.
        """
        priors = {
            # ctx 0 Drug efficacy → MeSH+RCT best
            (0,0):0.72, (0,1):0.55, (0,2):0.48, (0,3):0.52, (0,4):0.68,
            # ctx 1 Epidemiology → Keyword+Date best
            (1,0):0.52, (1,1):0.70, (1,2):0.45, (1,3):0.50, (1,4):0.58,
            # ctx 2 Mechanism → Author+Journal
            (2,0):0.50, (2,1):0.48, (2,2):0.68, (2,3):0.52, (2,4):0.55,
            # ctx 3 Treatment comparison → Systematic review best
            (3,0):0.58, (3,1):0.50, (3,2):0.45, (3,3):0.55, (3,4):0.72,
        }
        return priors.get((ctx, arm), 0.55)

    def get_state(self) -> np.ndarray:
        """Return (N_CONTEXTS × N_ARMS) mean reward matrix."""
        conn  = sqlite3.connect(self.db_path)
        rows  = conn.execute("SELECT context_id,arm_id,mean_reward FROM bandit_state").fetchall()
        conn.close()
        mat = np.full((self.N_CONTEXTS, self.N_ARMS), 0.5)
        for ctx, arm, mean in rows:
            if ctx < self.N_CONTEXTS and arm < self.N_ARMS:
                mat[ctx][arm] = mean
        return mat

    def select_arm(self, context_id: int, t: Optional[int] = None) -> tuple:
        """UCB arm selection for given context."""
        conn   = sqlite3.connect(self.db_path)
        rows   = conn.execute(
            "SELECT arm_id,mean_reward,n_pulls FROM bandit_state WHERE context_id=?",
            (context_id,)).fetchall()
        conn.close()
        if t is None:
            t = sum(r[2] for r in rows) + 1
        best_score = -1
        best_arm   = 0
        for arm_id, mean, n in rows:
            bonus = self.UCB_C * math.sqrt(math.log(t + 1) / max(n, 1))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_arm   = arm_id
        return best_arm, float(best_score)

    def update(self, context_id: int, arm_id: int, reward: float,
               session_id: str = "", query: str = "",
               top_grade: str = "", user_rating: Optional[int] = None):
        """Incremental mean update after observing reward."""
        conn = sqlite3.connect(self.db_path)
        row  = conn.execute(
            "SELECT mean_reward,n_pulls FROM bandit_state WHERE context_id=? AND arm_id=?",
            (context_id, arm_id)).fetchone()
        if row:
            mean, n = row
            n_new    = n + 1
            mean_new = mean + (reward - mean) / n_new
            conn.execute(
                "UPDATE bandit_state SET mean_reward=?,n_pulls=? WHERE context_id=? AND arm_id=?",
                (round(mean_new,4), n_new, context_id, arm_id))
        conn.execute(
            "INSERT INTO feedback VALUES(NULL,?,?,?,?,?,?,?,?)",
            (session_id, query[:200], arm_id, context_id, round(reward,4),
             user_rating, top_grade, time.time()))
        # Update session
        conn.execute(
            "INSERT INTO sessions VALUES(?,?,1) ON CONFLICT(session_id) "
            "DO UPDATE SET n_queries=n_queries+1",
            (session_id, time.time()))
        conn.commit()
        conn.close()

    def apply_rlhf(self, context_id: int, arm_id: int, rating: int):
        """
        Apply RLHF feedback (thumbs up=+1, thumbs down=-1).
        Shifts mean reward by RLHF_LR in direction of feedback.
        """
        delta = self.RLHF_LR * rating   # +0.15 or -0.15
        conn  = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE bandit_state SET mean_reward=MAX(0.05,MIN(1.0,mean_reward+?)) "
            "WHERE context_id=? AND arm_id=?",
            (round(delta,4), context_id, arm_id))
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Return aggregate stats for display."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rated = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_rating IS NOT NULL").fetchone()[0]
        thumbs_up   = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_rating=1").fetchone()[0]
        thumbs_down = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_rating=-1").fetchone()[0]
        sessions    = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        avg_reward  = conn.execute("SELECT AVG(reward) FROM feedback").fetchone()[0] or 0
        conn.close()
        return {
            "total_queries": total,
            "rated_queries":  rated,
            "thumbs_up":      thumbs_up,
            "thumbs_down":    thumbs_down,
            "sessions":       sessions,
            "avg_reward":     round(avg_reward, 4),
            "rlhf_rate":      round(rated/max(total,1)*100, 1),
        }

    def get_learning_curve(self, n_recent: int = 50) -> list:
        """Return recent reward trajectory for display."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT reward FROM feedback ORDER BY timestamp DESC LIMIT ?",
            (n_recent,)).fetchall()
        conn.close()
        rewards = [r[0] for r in reversed(rows)]
        # 5-point rolling average
        smoothed = []
        for i in range(len(rewards)):
            window = rewards[max(0,i-4):i+1]
            smoothed.append(round(sum(window)/len(window), 4))
        return smoothed
