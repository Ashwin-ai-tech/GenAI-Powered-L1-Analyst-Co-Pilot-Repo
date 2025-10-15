# new_backend.py
import os, glob, json, sqlite3, numpy as np, time, hashlib, re, uuid, logging
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq
import asyncio
import threading
from queue import Queue
import re
from database import get_db
from analytics_manager import AnalyticsManager

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_AVAILABLE = True
except Exception:
    CROSS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False


# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "llama-3.1-8b-instant")
BI_MODEL = "all-mpnet-base-v2"
CROSS_MODEL = os.getenv("CROSS_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
KB_GLOB = os.getenv("KB_GLOB", "./kb/**/*.json")
TOP_K = 8
BI_TOP_K = 25
DB_PATH = os.getenv("QUERY_DB", "query_history.sqlite")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "300"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
RELEVANCE_THRESHOLD = 0.35
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # <-- THIS IS THE FIX
logger = logging.getLogger(__name__)

# --- CRITICAL CHANGE: Initialize AnalyticsManager here ---
# It's now safe to do this because it no longer tries to access the DB in its __init__ method
analytics_manager = AnalyticsManager(DB_PATH)


# --- Data Structures & Global State ---
# (All your dataclasses and global variables like knowledge_base, documents, etc., remain here)
# ...
@dataclass
class PerformanceMetrics:
    query_time: float
    retrieval_time: float
    generation_time: float
    confidence: float
    cache_hit: bool
    kb_used: bool
    exact_match: bool
    query_length: int
    response_length: int
    used_context: bool = False
    clarification_asked: bool = False

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[Dict] = field(default_factory=list)
    pending_clarification: Optional[Dict] = None
    feedback_pending: bool = False
    session_title: str = "New Chat"
    theme_preference: str = "system"  # "light", "dark", or "system"

# -------------------------
# Global State
# -------------------------
performance_log: List[PerformanceMetrics] = []
query_cache = {}
knowledge_base, documents = [], []
prediction_cache = {}
prediction_queue = Queue()

# --- Database Migration & Setup ---
def migrate_database():
    """Add missing columns to existing tables"""
    try:
            con = get_db()
            cur = con.cursor()
            
            # Check sessions table structure
            cur.execute("PRAGMA table_info(sessions)")
            columns = [column[1] for column in cur.fetchall()]
            
            # Add missing columns to sessions table
            if 'feedback_pending' not in columns:
                logger.info("Adding feedback_pending column to sessions table...")
                cur.execute("ALTER TABLE sessions ADD COLUMN feedback_pending INTEGER DEFAULT 0")
            
            if 'session_title' not in columns:
                logger.info("Adding session_title column to sessions table...")
                cur.execute("ALTER TABLE sessions ADD COLUMN session_title TEXT DEFAULT 'New Chat'")
            
            if 'is_active' not in columns:
                logger.info("Adding is_active column to sessions table...")
                cur.execute("ALTER TABLE sessions ADD COLUMN is_active INTEGER DEFAULT 1")
            
            # Add theme_preference column if missing
            if 'theme_preference' not in columns:
                logger.info("Adding theme_preference column to sessions table...")
                cur.execute("ALTER TABLE sessions ADD COLUMN theme_preference TEXT DEFAULT 'system'")
            
            # Check history table structure
            cur.execute("PRAGMA table_info(history)")
            history_columns = [column[1] for column in cur.fetchall()]
            
            # Add missing columns to history table
            missing_history_columns = {
                'clarification_asked': 'INTEGER DEFAULT 0',
                'user_feedback': 'INTEGER',
                'feedback_comment': 'TEXT'
            }
            
            for col_name, col_type in missing_history_columns.items():
                if col_name not in history_columns:
                    logger.info(f"Adding {col_name} column to history table...")
                    cur.execute(f"ALTER TABLE history ADD COLUMN {col_name} {col_type}")
            
            con.commit()
            logger.info("Database migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration error: {e}")


def init_db():
        con = get_db()
        cur = con.cursor()
        # History table - COMPLETE SCHEMA
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                session_id TEXT, 
                query TEXT,
                answer TEXT, 
                confidence REAL, 
                used_kb INTEGER, 
                exact_match INTEGER,
                clarification_asked INTEGER DEFAULT 0,
                user_feedback INTEGER,
                feedback_comment TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Feedback table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                session_id TEXT, 
                query TEXT,
                answer TEXT, 
                rating INTEGER, 
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Sessions table - COMPLETE SCHEMA
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, 
                user_id TEXT, 
                created_at DATETIME,
                last_activity DATETIME, 
                conversation_history TEXT, 
                pending_clarification TEXT,
                feedback_pending INTEGER DEFAULT 0,
                session_title TEXT DEFAULT 'New Chat',
                is_active INTEGER DEFAULT 1,
                theme_preference TEXT DEFAULT 'system'
            )
        """)
        # Analytics table (simplified - detailed analytics moved to analytics_manager)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_queries INTEGER DEFAULT 0,
                positive_feedback INTEGER DEFAULT 0,
                negative_feedback INTEGER DEFAULT 0,
                kb_usage_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.commit()
    
    # Run migration to ensure existing databases are updated
        migrate_database()



# -------------------------
# Next Word Prediction System
# -------------------------
class NextWordPredictor:
    def __init__(self):
        self.common_phrases = [
            "How do I troubleshoot", "How to fix", "Error when", "Problem with",
            "Cannot connect to", "Login issue", "Password reset", "Network connectivity",
            "Server not responding", "Application crashing", "Performance slow",
            "File not found", "Access denied", "Configuration issue", "Installation problem"
        ]
        self.tech_keywords = [
            "server", "database", "network", "application", "system", "error", "issue",
            "problem", "fix", "solution", "troubleshoot", "configure", "install",
            "update", "restart", "permission", "access", "login", "password"
        ]
    
    def predict_next_words(self, partial_text: str, max_predictions: int = 3) -> List[str]:
        """Predict next words based on partial input"""
        if not partial_text.strip():
            return self.common_phrases[:max_predictions]
        
        partial_lower = partial_text.lower().strip()
        predictions = []
        
        # Match with common phrases
        for phrase in self.common_phrases:
            if phrase.lower().startswith(partial_lower) and phrase != partial_text:
                predictions.append(phrase)
        
        # Add technical keywords if relevant
        if len(predictions) < max_predictions:
            for keyword in self.tech_keywords:
                if keyword not in partial_lower and len(predictions) < max_predictions:
                    predictions.append(f"{partial_text} {keyword}")
        
        return predictions[:max_predictions]

next_word_predictor = NextWordPredictor()

def get_next_word_predictions(partial_text: str) -> List[str]:
    """Get real-time next word predictions"""
    cache_key = hash(partial_text.lower().strip())
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]
    
    predictions = next_word_predictor.predict_next_words(partial_text)
    prediction_cache[cache_key] = predictions
    
    # Limit cache size
    if len(prediction_cache) > 100:
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]
    
    return predictions

# -------------------------
# Enhanced Session Management with Theme Support
# -------------------------
def get_session(session_id: str, user_id: str = "default") -> Session:
    """Get or create session from database"""
    con = get_db()
    cur = con.cursor()
    cur.execute("""
            SELECT user_id, created_at, last_activity, conversation_history, 
                   pending_clarification, feedback_pending, session_title, theme_preference
            FROM sessions WHERE session_id = ? AND is_active = 1
        """, (session_id,))
    row = cur.fetchone()

    now = datetime.now()
    if row:
            db_user_id, created_at_iso, last_activity_iso, history_json, clarification_json, feedback_pending, session_title, theme_preference = row
            last_activity = datetime.fromisoformat(last_activity_iso)
            
            if now - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                # Session expired, create new one with same ID but fresh history
                return Session(
                    session_id=session_id, user_id=user_id, created_at=now,
                    last_activity=now, conversation_history=[], 
                    pending_clarification=None, feedback_pending=False,
                    session_title="New Chat", theme_preference=theme_preference or "system"
                )
            
            return Session(
                session_id=session_id, user_id=db_user_id, 
                created_at=datetime.fromisoformat(created_at_iso),
                last_activity=now, 
                conversation_history=json.loads(history_json or '[]'),
                pending_clarification=json.loads(clarification_json) if clarification_json else None,
                feedback_pending=bool(feedback_pending),
                session_title=session_title or "New Chat",
                theme_preference=theme_preference or "system"
            )
    else:
            # Create new session
            session_title = "New Chat"
            return Session(
                session_id=session_id, user_id=user_id, created_at=now,
                last_activity=now, conversation_history=[], 
                pending_clarification=None, feedback_pending=False,
                session_title=session_title, theme_preference="system"
            )

def save_session(session: Session):
    """Save session to database"""
    con = get_db()
    cur = con.cursor()
    history_json = json.dumps(session.conversation_history)
    clarification_json = json.dumps(session.pending_clarification) if session.pending_clarification else None
        
        # Auto-generate session title from first user message if not set
    session_title = session.session_title
    if session_title == "New Chat" and session.conversation_history:
            first_user_msg = next((msg.get("query", "") for msg in session.conversation_history if msg.get("role") != "assistant"), "")
            if first_user_msg:
                session_title = first_user_msg[:40] + "..." if len(first_user_msg) > 40 else first_user_msg
        
    cur.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, user_id, created_at, last_activity, conversation_history, 
             pending_clarification, feedback_pending, session_title, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            session.session_id, session.user_id, session.created_at.isoformat(),
            session.last_activity.isoformat(), history_json, clarification_json,
            int(session.feedback_pending), session_title, session.theme_preference
        ))
    con.commit()

def update_theme_preference(session_id: str, theme: str, user_id: str = "default") -> bool:
    """Update user's theme preference in their session"""
    try:
        session = get_session(session_id, user_id)
        session.theme_preference = theme
        save_session(session)
        return True
    except Exception as e:
        logger.error(f"Error updating theme preference: {e}")
        return False

def get_theme_preference(session_id: str, user_id: str = "default") -> str:
    """Get user's theme preference from their session"""
    try:
        session = get_session(session_id, user_id)
        return session.theme_preference
    except Exception as e:
        logger.error(f"Error getting theme preference: {e}")
        return "system"  # Default to system preference

def get_all_sessions_from_db(user_id: str = "default") -> List[Dict]:
    """Get all active sessions for sidebar with full conversation data"""
    sessions_data = []
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                SELECT session_id, session_title, created_at, last_activity, conversation_history, theme_preference
                FROM sessions 
                WHERE user_id = ? AND is_active = 1
                ORDER BY last_activity DESC
            """, (user_id,))
            
            rows = cur.fetchall()
            for session_id, session_title, created_at, last_activity, history_json, theme_preference in rows:
                # Parse conversation history to get message count
                conversation_history = json.loads(history_json or '[]')
                message_count = len(conversation_history)
                
                sessions_data.append({
                    "session_id": session_id,
                    "title": session_title or "New Chat",
                    "created_at": created_at,
                    "last_activity": last_activity,
                    "message_count": message_count,
                    "theme_preference": theme_preference or "system",
                    "conversation_history": conversation_history  # Include full history for continuation
                })
                
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
    
    return sessions_data

def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict]:
    """
    Get conversation history for frontend compatibility
    This bridges Version 1 backend with Version 2 frontend
    """
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                SELECT query, answer, confidence, used_kb, exact_match, 
                       clarification_asked, timestamp 
                FROM history 
                WHERE session_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """, (session_id, limit))
            
            rows = cur.fetchall()
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "answer": row[1],
                    "confidence": float(row[2]),
                    "used_kb": bool(row[3]),
                    "exact_match": bool(row[4]),
                    "clarification_asked": bool(row[5]),
                    "timestamp": row[6]
                })
            return history
            
    except Exception as e:
        logger.error(f"Error getting conversation history for session {session_id}: {e}")
        return []

def create_new_session(user_id: str = "default") -> Dict[str, Any]:
    """Create a new session and return session info"""
    session_id = str(uuid.uuid4())[:12]
    session = Session(
        session_id=session_id,
        user_id=user_id,
        created_at=datetime.now(),
        last_activity=datetime.now(),
        session_title="New Chat",
        feedback_pending=False,  # Explicitly set to avoid issues
        theme_preference="system"
    )
    save_session(session)
    
    return {
        "session_id": session_id,
        "title": "New Chat",
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "message_count": 0,
        "theme_preference": "system",
        "conversation_history": []
    }

def delete_session(session_id: str, user_id: str = "default") -> bool:
    """Soft delete a session (mark as inactive)"""
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                UPDATE sessions SET is_active = 0 
                WHERE session_id = ? AND user_id = ?
            """, (session_id, user_id))
            
            if cur.rowcount > 0:
                con.commit()
                logger.info(f"Session {session_id} deleted by user {user_id}")
                return True
            else:
                return False
                
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return False

def rename_session(session_id: str, new_title: str, user_id: str = "default") -> bool:
    """Rename a session"""
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                UPDATE sessions SET session_title = ?
                WHERE session_id = ? AND user_id = ? AND is_active = 1
            """, (new_title, session_id, user_id))
            
            if cur.rowcount > 0:
                con.commit()
                return True
            else:
                return False
                
    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {e}")
        return False

def load_session(session_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
    """Load a specific session with full conversation history"""
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                SELECT session_id, session_title, created_at, last_activity, conversation_history, theme_preference
                FROM sessions 
                WHERE session_id = ? AND user_id = ? AND is_active = 1
            """, (session_id, user_id))
            
            row = cur.fetchone()
            if row:
                session_id, session_title, created_at, last_activity, history_json, theme_preference = row
                conversation_history = json.loads(history_json or '[]')
                
                return {
                    "session_id": session_id,
                    "title": session_title or "New Chat",
                    "created_at": created_at,
                    "last_activity": last_activity,
                    "message_count": len(conversation_history),
                    "theme_preference": theme_preference or "system",
                    "conversation_history": conversation_history
                }
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None

def clear_session_history(session_id: str, user_id: str = "default") -> bool:
    """Clear conversation history for a session but keep the session"""
    try:
        session = get_session(session_id, user_id)
        session.conversation_history = []
        session.feedback_pending = False
        session.pending_clarification = None
        session.session_title = "New Chat"
        save_session(session)
        return True
    except Exception as e:
        logger.error(f"Error clearing session history {session_id}: {e}")
        return False

def get_session_manager(user_id: str = "default") -> Dict[str, Any]:
    """Get comprehensive session management data"""
    sessions = get_all_sessions_from_db(user_id)
    current_session = create_new_session(user_id) if not sessions else sessions[0]
    
    return {
        "sessions": sessions,
        "current_session": current_session,
        "total_sessions": len(sessions)
    }

def switch_session(session_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
    """Switch to a different session"""
    return load_session(session_id, user_id)

# -------------------------
# Knowledge Base Loading
# -------------------------
def load_knowledge_base(kb_glob: str = KB_GLOB) -> Tuple[List[Dict], List[str]]:
    """Load and index knowledge base"""
    files = glob.glob(kb_glob, recursive=True)
    entries, texts = [], []
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    # Create enhanced content for better retrieval
                    title = item.get("title", "")
                    solution = item.get("solution_answer", "")
                    steps = item.get("step_by_step_instructions", [])
                    
                    enhanced_content = f"{title}. {solution}"
                    if steps and isinstance(steps, list):
                        enhanced_content += " Steps: " + ". ".join(steps)
                    
                    if enhanced_content.strip():
                        entry = {
                            "title": title,
                            "article_number": item.get("article_number", "") or item.get("id", "") or "no-id",
                            "content": enhanced_content,
                            "solution_answer": solution,
                            "step_by_step_instructions": steps,
                            "raw": item,
                            "source_file": os.path.basename(f)
                        }
                        entries.append(entry)
                        texts.append(enhanced_content)
                        
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
    
    logger.info(f"Loaded {len(entries)} KB entries from {len(files)} files.")
    return entries, texts

# Load knowledge base
knowledge_base, documents = load_knowledge_base(KB_GLOB)

# -------------------------
# Embeddings and Search Index
# -------------------------
logger.info(f"Loading bi-encoder ({BI_MODEL})...")
bi_encoder = SentenceTransformer(BI_MODEL)

if documents:
    doc_embeddings = bi_encoder.encode(documents, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    doc_embeddings_norm = doc_embeddings / norms
    dim = doc_embeddings_norm.shape[1]
    
    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings_norm)
    logger.info(f"Indexed {index.ntotal} vectors (dim={dim}).")
else:
    doc_embeddings, doc_embeddings_norm, index = np.zeros((0,1), dtype="float32"), np.array([]), None

# BM25 setup
if BM25_AVAILABLE and documents:
    tokenized_docs = [re.findall(r'\w+', d.lower()) for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
else:
    bm25 = None

# Cross-encoder setup
if CROSS_AVAILABLE:
    try:
        logger.info(f"Loading cross-encoder ({CROSS_MODEL})...")
        cross_encoder = CrossEncoder(CROSS_MODEL)
    except Exception as e:
        cross_encoder, CROSS_AVAILABLE = None, False
        logger.error(f"Failed to load cross-encoder: {e}")
else:
    cross_encoder = None

# -------------------------
# Enhanced Retrieval System
# -------------------------
def improved_hybrid_retrieve(query: str, bi_top_k: int = BI_TOP_K, top_k: int = TOP_K) -> Tuple[List[Dict], Dict]:
    """Enhanced retrieval with multiple strategies"""
    if not documents or index is None:
        return [], {"confidence": 0.0, "used_kb": False, "exact_match": False}

    start_time = time.time()
    
    # Vector search
    q_emb = bi_encoder.encode([query], convert_to_numpy=True).astype("float32")
    qnorm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    qnorm[qnorm==0] = 1.0
    q_emb_norm = (q_emb / qnorm)[0]

    D, I = index.search(np.expand_dims(q_emb_norm, axis=0), bi_top_k)
    vector_indices = [int(i) for i in I[0] if i >= 0]
    vector_scores = [float(s) for s in D[0] if s >= 0]

    # BM25 search
    bm25_results = []
    if bm25 is not None:
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:bi_top_k]
        bm25_results = [{
            "idx": int(idx),
            "score": float(bm25_scores[idx])
        } for idx in bm25_indices if idx < len(documents)]

    # Reciprocal Rank Fusion
    def reciprocal_rank_fusion(vector_indices, vector_scores, bm25_results, k=60):
        scores = {}
        
        for rank, (idx, score) in enumerate(zip(vector_indices, vector_scores)):
            scores[idx] = scores.get(idx, 0) + (1 / (rank + k)) * score
        
        for rank, result in enumerate(bm25_results):
            idx = result["idx"]
            score = result["score"]
            scores[idx] = scores.get(idx, 0) + (1 / (rank + k)) * score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Get fused results
    fused_results = reciprocal_rank_fusion(vector_indices, vector_scores, bm25_results)
    
    # Apply relevance threshold
    filtered_results = [(idx, score) for idx, score in fused_results if score >= RELEVANCE_THRESHOLD]
    
    if not filtered_results:
        filtered_results = fused_results[:top_k]
    
    final_indices = [idx for idx, score in filtered_results[:top_k]]

    # Prepare candidates
    candidates = []
    for idx in final_indices:
        if idx >= len(knowledge_base):
            continue
            
        kb = knowledge_base[idx].copy()
        candidate = {
            "idx": idx,
            "title": kb.get("title", ""),
            "article_number": kb.get("article_number", "no-id"),
            "content": kb.get("content", ""),
            "solution_answer": kb.get("solution_answer", ""),
            "step_by_step_instructions": kb.get("step_by_step_instructions", []),
            "final_score": next((score for i, score in filtered_results if i == idx), 0.0),
        }
        candidates.append(candidate)

    # Cross-encoder re-ranking
    if cross_encoder is not None and candidates:
        pairs = [(query, c["content"][:1024]) for c in candidates]
        try:
            cross_scores = cross_encoder.predict(pairs)
            for i, score in enumerate(cross_scores):
                candidates[i]["cross_score"] = float(score)
                candidates[i]["final_score"] = candidates[i]["final_score"] * 0.7 + float(score) * 0.3
        except Exception as e:
            logger.error(f"Cross-encoder failed: {e}")

    # Sort by final score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Calculate confidence
    cross_arr = np.array([c.get("cross_score", 0.0) for c in candidates], dtype=float)
    top_prob = softmax_conf(cross_arr) if len(cross_arr) > 0 else 0.0
    exact_found = any(is_exact_match(query, knowledge_base[c["idx"]]) for c in candidates)
    used_kb = exact_found or (len(candidates) > 0 and top_prob >= 0.45)
    confidence = max(min(top_prob if not exact_found else 0.95, 0.999), 0.0)

    retrieval_time = time.time() - start_time
    metadata = {
        "confidence": confidence,
        "used_kb": used_kb,
        "exact_match": exact_found,
        "retrieval_time": retrieval_time,
        "candidates_found": len(candidates)
    }
    
    return candidates, metadata

def softmax_conf(scores: np.ndarray) -> float:
    if len(scores)==0: return 0.0
    s = scores - np.max(scores)
    ex = np.exp(s)
    probs = ex / (np.sum(ex)+1e-12)
    return float(np.max(probs))

def is_exact_match(query: str, kb_entry: Dict) -> bool:
    q = query.strip().lower()
    title = (kb_entry.get("title") or "").lower()
    return q in title or any(q in step.lower() for step in kb_entry.get("step_by_step_instructions", []) if isinstance(step, str))

# -------------------------
# LLM Integration
# -------------------------
def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1, max_tokens: int = 800) -> str:
    """Call Groq API for LLM responses"""
    if not GROQ_API_KEY:
        return "LLM service is currently unavailable. Please try again later."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."

# -------------------------
# Prompt Templates
# -------------------------
PROMPT_KB_TEMPLATE = """
**Persona:** You are "Astra," an advanced AI co-pilot for Production support. Your personality is professional, calm, confident, and highly empathetic. Your primary goal is to make the user feel heard and expertly guided toward a solution.

---
**Context Analysis:**
* **Latest User Message:** "{query}"
* **Knowledge Base Search Results (Your ONLY source of truth):**
    {knowledge_base_content}

---
**Strict Rules of Engagement:**

1.  **Stay in Character:** You are Astra. Begin your response directly, as if you are typing in a chat window.
2.  **Acknowledge First, Then Solve:** ALWAYS start by acknowledging the user's issue in your own words. This shows you are listening.
    * *Good Example:* "I'm sorry to hear you're running into a permissions error. Let's see what we can do about that."
    * *Bad Example:* "Here are the steps to fix permissions."
3.  **NEVER Reveal Your Process:** Do NOT say "Based on the knowledge base...", "I found an article...", or "The information says...". This breaks the illusion. Just provide the answer as if you already know it.
4.  **Be an Expert, Not a Document Reader:** Do not just copy-paste the KB. Synthesize the information into a clear, easy-to-read, conversational response. Use formatting like lists and bold text to improve clarity.
5.  **Provide a Clear Path Forward:** End your response by telling the user what to do next, even if it's just to try the solution.

---
**Task:** Generate the next response in the conversation.
"""

PROMPT_GENERAL_TEMPLATE = """
**Persona:** You are "Astra," an advanced AI co-pilot. You are acting as a mentor to an L1 IT support analyst. Your tone should be calm, confident, and supportive.

**Situation Analysis:**
* **User's Query:** "{query}"
* **Knowledge Base Status:** No specific documented procedure was found for this query.
* **Recent Conversation:**
    {conversation_context}

---
**Your Task:**
Your primary goal is to provide general, safe, and logical troubleshooting guidance without inventing a specific solution. You must be honest that you don't have a KB article for this.

**Follow these steps for your response:**

1.  **Acknowledge and State the Situation Gracefully:** Start by clearly and calmly stating that you don't have a specific procedure, but you can still help.
    * *Good Example:* "I don't have a specific knowledge base article for that exact issue, but we can definitely work through it with some general troubleshooting steps."
    * *Bad Example:* "I could not find an answer."

2.  **Provide General, Actionable Advice:** Based on the query, suggest 2-3 universal first steps an L1 analyst should take for that *type* of problem (e.g., for a database issue, suggest checking connection strings, service status, and network connectivity).

3.  **Ask for More Information:** Guide the user on what to look for. Ask clarifying questions that would help a human expert diagnose the issue. (e.g., "Are there any specific error codes in the logs?", "When did this issue start happening?", "Is this affecting a single user or multiple users?").

4.  **Maintain Your Persona:** End on a supportive and encouraging note.

**Strict Rules:**
-   DO NOT invent a multi-step solution. Provide general advice only.
-   DO NOT pretend you have a KB article. Honesty is key.
"""

PROMPT_CONVERSATIONAL_TEMPLATE = """
**Persona:** You are "Astra," a professional but personable IT support co-pilot. Your personality is friendly but focused on the task at hand.

**Core Objective:** Your goal in a conversational exchange is to be pleasant and quickly guide the user back to solving a technical problem. Keep your responses concise.

**User's Message:** "{query}"

---
**Your Task:**
Based on the user's message, choose the appropriate response type from the rules below.

**Response Playbook:**

1.  **If the user provides a GREETING (e.g., "hello", "hi"):**
    * Respond with a warm but professional greeting and immediately ask how you can help.
    * *Example:* "Hello! How can I assist with your technical issue today?"

2.  **If the user expresses GRATITUDE (e.g., "thank you", "thanks"):**
    * Acknowledge their thanks politely and ask if there is anything else they need.
    * *Example:* "You're very welcome! Is there anything else I can help you with?"

3.  **If the user asks a PERSONAL QUESTION (e.g., "how are you?"):**
    * Give a brief, friendly, non-human answer and immediately pivot back to your purpose.
    * *Example:* "I'm an AI, so I'm always running at optimal parameters! What technical problem can I help you solve?"

4.  **For any other casual statement:**
    * Provide a brief, positive acknowledgment and gently guide the conversation back to work.
    * *Example:* "Understood. Let me know when you have a specific technical query."

---
**Strict Rule:** Your primary purpose is IT support. Do not engage in long, non-technical conversations.
"""

# -------------------------
# Response Generation
# -------------------------

def build_kb_content_for_prompt(candidates: List[Dict]) -> str:
    """Formats the top KB candidates into a single string for the prompt."""
    if not candidates:
        return "No relevant knowledge base content was found."
    
    content_parts = []
    for i, candidate in enumerate(candidates[:3]): # Use top 3 candidates
        title = candidate.get("title", "Untitled")
        content = candidate.get("content", "No content.")
        content_parts.append(f"Article {i+1}: {title}\nContent: {content}")
        
    return "\n\n---\n\n".join(content_parts)

import re

def simple_pii_scrubber(text: str) -> str:
    """
    A lightweight PII scrubber using regular expressions.
    """
    # Regex for email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL_ADDRESS>', text)
    
    # Regex for phone numbers (matches various common formats)
    text = re.sub(r'(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}', '<PHONE_NUMBER>', text)
    
    # Regex for IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP_ADDRESS>', text)
    
    return text


def generate_kb_response(query: str, candidates: List[Dict]) -> str:
    """
    Generate response FROM knowledge base content BY USING THE LLM.
    """
    # 1. Format the retrieved KB content into a string for the prompt.
    knowledge_base_content = build_kb_content_for_prompt(candidates)
    
    # 2. Anonymize the KB content using the LIGHTWEIGHT scrubber.
    scrubbed_kb_content = simple_pii_scrubber(knowledge_base_content) # <<< YOUR NEW LIGHTWEIGHT STEP
    
    # 3. Format your master prompt with the query and the SCRUBBED KB content.
    prompt = PROMPT_KB_TEMPLATE.format(
        query=query,
        knowledge_base_content=scrubbed_kb_content # Use the scrubbed version
    )
    
    # 4. Call the LLM with the complete, secure prompt.
    response = call_llm(prompt)
    
    return response

def generate_general_response(query: str, conversation_context: str = "") -> str:
    """Generate response using LLM when KB doesn't have answer"""
    prompt = PROMPT_GENERAL_TEMPLATE.format(
        query=query,
        conversation_context=conversation_context or "No previous context"
    )
    return call_llm(prompt)

def generate_conversational_response(query: str) -> str:
    """Generate response for casual conversation"""
    prompt = PROMPT_CONVERSATIONAL_TEMPLATE.format(query=query)
    return call_llm(prompt, temperature=0.3)

# -------------------------
# Enhanced Intent Clarification System
# -------------------------
def is_vague_query(query: str) -> Tuple[bool, Optional[str]]:
    """Detect vague queries and return (is_vague, clarification_prompt)"""
    query_lower = query.lower().strip()
    
    # Common vague patterns
    vague_patterns = {
        "it": "Could you specify what 'it' refers to? What exactly isn't working?",
        "this": "What specific issue are you having with 'this'? Please describe the problem in more detail.",
        "that": "Could you tell me more about 'that'? What specific function or feature is causing problems?",
        "something": "What exactly isn't working properly? Please describe the specific issue.",
        "thing": "Which specific thing are you referring to? Please name the component or function.",
        "problem": "What specific problem are you experiencing? Please describe the error or issue.",
        "issue": "Could you describe the exact issue you're facing? What error messages are you seeing?",
        "error": "What specific error are you encountering? Please share the exact error message.",
        "help": "What specific area do you need help with? Please describe your technical issue.",
        "not working": "What exactly isn't working? Please specify the application, feature, or service."
    }
    
    # Check for short, nonspecific queries
    if len(query.split()) <= 2:
        return True, "Could you please provide more details about your technical issue? What specifically are you trying to do or fix?"
    
    # Check for vague pronouns and terms
    for vague_term, clarification in vague_patterns.items():
        if vague_term in query_lower and len(query.split()) <= 4:
            return True, clarification
    
    # Check for lack of specific technical context
    technical_terms = {'server', 'database', 'network', 'application', 'login', 'password', 
                      'email', 'connection', 'install', 'update', 'configure', 'error', 'crash'}
    if not any(term in query_lower for term in technical_terms):
        if len(query.split()) < 5:  # Very short, non-technical queries
            return True, "Are you having a technical issue? Please describe what you're trying to accomplish or what problem you're experiencing."
    
    return False, None

def is_conversational_query(query: str) -> bool:
    """Check if query is conversational (greetings, thanks, etc.)"""
    conversational_phrases = {
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'thanks', 'thank you', 'thank', 'thx', 'appreciate it',
        'bye', 'goodbye', 'see you', 'later',
        'yes', 'no', 'ok', 'okay', 'sure', 'got it', 'understand',
        'how are you', 'what\'s up', 'hey there'
    }
    
    query_lower = query.lower().strip()
    return any(phrase in query_lower for phrase in conversational_phrases)

def handle_conversational_query(query: str) -> str:
    """Handle conversational queries with appropriate responses"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(phrase in query_lower for phrase in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return "Hello! I'm Astra, your IT support assistant. How can I help you with your technical issues today?"
    
    # Thanks
    if any(phrase in query_lower for phrase in ['thanks', 'thank you', 'thx', 'appreciate']):
        return "You're welcome! I'm glad I could help. Is there anything else you need assistance with?"
    
    # Farewell
    if any(phrase in query_lower for phrase in ['bye', 'goodbye', 'see you', 'later']):
        return "Goodbye! Feel free to reach out if you have any other IT questions."
    
    # Default conversational response
    return generate_conversational_response(query)

# -------------------------
# Main Query Function with Enhanced Session Support and Intent Clarification
# -------------------------
def query_rag(query: str, session_id: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
    """Main function to handle user queries with full conversation flow and session management"""
    total_start = time.time()
    
    # Create new session if none provided
    if not session_id:
        new_session = create_new_session(user_id)
        session_id = new_session["session_id"]
    
    # Get or create session
    session = get_session(session_id, user_id)
    
    # Handle conversational queries
    if is_conversational_query(query):
        response = handle_conversational_query(query)
        session.conversation_history.append({
            "query": query, 
            "answer": response, 
            "timestamp": datetime.now().isoformat(),
            "used_kb": False,
            "confidence": 0.0
        })
        save_session(session)
        
        return {
            "answer": response,
            "confidence": 0.0,
            "used_kb": False,
            "exact_match": False,
            "session_id": session.session_id,
            "feedback_required": False,
            "total_time": time.time() - total_start,
            "session_title": session.session_title,
            "theme_preference": session.theme_preference
        }
    
    # Check for vague queries
    is_vague, clarification_prompt = is_vague_query(query)
    if is_vague:
        # Store the pending clarification in session
        session.pending_clarification = {
            "original_query": query,
            "clarification_prompt": clarification_prompt,
            "timestamp": datetime.now().isoformat()
        }
        save_session(session)
        
        return {
            "answer": clarification_prompt,
            "confidence": 0.1,
            "used_kb": False,
            "exact_match": False,
            "session_id": session.session_id,
            "feedback_required": False,
            "clarification_asked": True,
            "total_time": time.time() - total_start,
            "session_title": session.session_title,
            "theme_preference": session.theme_preference
        }
    
    # Handle pending clarifications
    if session.pending_clarification:
        # Combine the original query with the clarification response
        enhanced_query = f"{session.pending_clarification['original_query']} - User provided: {query}"
        query = enhanced_query
        session.pending_clarification = None
    
    # Handle technical queries
    candidates, metadata = improved_hybrid_retrieve(query)
    
    if metadata.get("used_kb", False) and candidates:
        # Generate response from knowledge base
        response = generate_kb_response(query, candidates)
        confidence = metadata["confidence"]
        used_kb = True
    else:
        # Generate general response using LLM
        conversation_context = build_conversation_context(session)
        response = generate_general_response(query, conversation_context)
        confidence = 0.3
        used_kb = False
    
    # Add feedback prompt for KB-based answers
    feedback_required = used_kb and confidence > 0.5
    if feedback_required:
        response += "\n\n---\n*Was this answer helpful? Please provide feedback with thumbs up/down.*"
        session.feedback_pending = True
    
    # Update session
    session.conversation_history.append({
        "query": query,
        "answer": response,
        "timestamp": datetime.now().isoformat(),
        "used_kb": used_kb,
        "confidence": confidence
    })
    
    # Auto-update session title from first meaningful query
    if session.session_title == "New Chat" and len(session.conversation_history) >= 2:
        first_user_msg = session.conversation_history[0].get("query", "")
        if first_user_msg and not is_conversational_query(first_user_msg):
            session.session_title = first_user_msg[:40] + "..." if len(first_user_msg) > 40 else first_user_msg
    
    # Limit history size
    if len(session.conversation_history) > 50:
        session.conversation_history = session.conversation_history[-50:]
    
    save_session(session)
    
    # Log to history table
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                INSERT INTO history (session_id, query, answer, confidence, used_kb, exact_match)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session.session_id, query, response, confidence, int(used_kb), int(metadata.get("exact_match", False))))
            con.commit()
    except Exception as e:
        logger.error(f"Error logging to history: {e}")
    
    return {
        "answer": response,
        "confidence": confidence,
        "used_kb": used_kb,
        "exact_match": metadata.get("exact_match", False),
        "session_id": session.session_id,
        "feedback_required": feedback_required,
        "total_time": time.time() - total_start,
        "retrieval_time": metadata.get("retrieval_time", 0),
        "session_title": session.session_title,
        "theme_preference": session.theme_preference
    }

def build_conversation_context(session: Session, limit: int = 5) -> str:
    """Build conversation context for LLM"""
    if not session.conversation_history:
        return ""
    
    context_lines = []
    for turn in session.conversation_history[-limit:]:
        context_lines.append(f"User: {turn.get('query', '')}")
        context_lines.append(f"Assistant: {turn.get('answer', '')[:100]}...")
    
    return "\n".join(context_lines)

# -------------------------
# Feedback System (Simplified - uses AnalyticsManager)
# -------------------------
def submit_feedback(session_id: str, rating: int, comment: str = "") -> Dict[str, Any]:
    """Submit user feedback using AnalyticsManager"""
    try:
            con = get_db()
            cur = con.cursor()
            
            # Get last interaction
            cur.execute("""
                SELECT query, answer FROM history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (session_id,))
            row = cur.fetchone()
            
            if row:
                query, answer = row
                # Store feedback using AnalyticsManager
                analytics_manager.record_feedback(session_id, query, answer, rating, comment)
                
                # Update session feedback status
                cur.execute("""
                    UPDATE sessions SET feedback_pending = 0 WHERE session_id = ?
                """, (session_id,))
                
                con.commit()
            
            # Get updated analytics
            analytics = analytics_manager.get_comprehensive_analytics()
            return {"success": True, "analytics": analytics}
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {"success": False, "error": str(e)}

# -------------------------
# Performance Monitoring (Simplified - uses AnalyticsManager)
# -------------------------
def get_performance_stats() -> Dict[str, Any]:
    """Get system performance statistics using AnalyticsManager"""
    return analytics_manager.get_performance_stats()


# -------------------------
# Utility Functions
# -------------------------
def get_last_turns(limit: int = 5) -> List[Dict]:
    """Get recent conversation turns across all sessions"""
    try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                SELECT session_id, query, answer, confidence, used_kb, timestamp
                FROM history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "session_id": row[0],
                    "query": row[1],
                    "answer": row[2],
                    "confidence": row[3],
                    "used_kb": bool(row[4]),
                    "timestamp": row[5]
                }
                for row in cur.fetchall()
            ]
    except Exception as e:
        logger.error(f"Error getting last turns: {e}")
        return []

def diagnose_retrieval(query: str) -> Dict[str, Any]:
    """Diagnostic function for retrieval system"""
    candidates, metadata = improved_hybrid_retrieve(query)
    
    return {
        "query": query,
        "candidates_found": len(candidates),
        "confidence": metadata.get("confidence", 0),
        "used_kb": metadata.get("used_kb", False),
        "top_candidates": [
            {
                "title": c.get("title", ""),
                "score": round(c.get("final_score", 0), 3),
                "content_preview": c.get("content", "")[:100] + "..."
            }
            for c in candidates[:3]
        ]
    }
