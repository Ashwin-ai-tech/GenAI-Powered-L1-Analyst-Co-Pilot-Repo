
import os, glob, json, sqlite3, numpy as np, time, hashlib, re, uuid
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv

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

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# -------------------------
# Config
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
BI_MODEL = os.getenv("BI_MODEL", "all-MiniLM-L6-v2")
CROSS_MODEL = os.getenv("CROSS_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
KB_GLOB = os.getenv("KB_GLOB", "./knowledge-bases/**/*.json")
TOP_K = int(os.getenv("TOP_K", "3"))
BI_TOP_K = int(os.getenv("BI_TOP_K", "15"))
DB_PATH = os.getenv("QUERY_DB", "query_history.sqlite")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "300"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

# Configure Gemini
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try: 
        genai.configure(api_key=GEMINI_API_KEY)
    except: 
        pass

# -------------------------
# Data Structures
# -------------------------
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
    conversation_history: List[Dict]
    pending_clarification: Optional[Dict] = None

# -------------------------
# Global State
# -------------------------
performance_log: List[PerformanceMetrics] = []
query_cache = {}
sessions: Dict[str, Session] = {}
knowledge_base, documents = [], []

# -------------------------
# SQLite helpers
# -------------------------
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = get_db_connection()
    cur = con.cursor()
    
    # History table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            answer TEXT,
            confidence REAL,
            used_kb INTEGER,
            exact_match INTEGER,
            clarification_asked INTEGER,
            top_k_ids TEXT,
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
    
    # Sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at DATETIME,
            last_activity DATETIME,
            conversation_history TEXT,
            pending_clarification TEXT
        )
    """)
    
    con.commit()
    con.close()

init_db()

# -------------------------
# Session Management
# -------------------------
def create_session(user_id: str = "default") -> Session:
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        user_id=user_id,
        created_at=datetime.now(),
        last_activity=datetime.now(),
        conversation_history=[],
        pending_clarification=None
    )
    sessions[session_id] = session
    return session

def get_session(session_id: Optional[str] = None, user_id: str = "default") -> Session:
    """Get or create a session"""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        # Check if session expired
        if datetime.now() - session.last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            # Session expired, create new one
            return create_session(user_id)
        session.last_activity = datetime.now()
        return session
    
    return create_session(user_id)

def save_session_to_db(session: Session):
    """Save session to database for persistence"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Convert conversation history to JSON
        history_json = json.dumps(session.conversation_history)
        clarification_json = json.dumps(session.pending_clarification) if session.pending_clarification else None
        
        cur.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, user_id, created_at, last_activity, conversation_history, pending_clarification)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.user_id,
            session.created_at.isoformat(),
            session.last_activity.isoformat(),
            history_json,
            clarification_json
        ))
        
        con.commit()
        con.close()
    except Exception as e:
        print(f"❌ Error saving session to DB: {e}")

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    expired_sessions = []
    now = datetime.now()
    
    for session_id, session in list(sessions.items()):
        if now - session.last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del sessions[session_id]

# -------------------------
# Knowledge Base Loading
# -------------------------
def smart_chunking(content: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Smart chunking that preserves semantic boundaries"""
    if not content or len(content.split()) <= max_chunk_size:
        return [content]
    
    chunks = []
    sentences = [s.strip() for s in content.split('. ') if s.strip()]
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        if current_word_count + sentence_word_count > max_chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def load_kbs_with_smart_chunking(kb_glob: str = KB_GLOB) -> Tuple[List[Dict], List[str]]:
    """Load knowledge bases with smart chunking"""
    files = glob.glob(kb_glob, recursive=True)
    entries, texts = [], []
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    content = (
                        (item.get("content") if isinstance(item.get("content"), str) else None)
                        or (item.get("text") if isinstance(item.get("text"), str) else None)
                        or (item.get("solution_answer") if isinstance(item.get("solution_answer"), str) else None)
                        or (item.get("summary") if isinstance(item.get("summary"), str) else None)
                        or " ".join(item.get("step_by_step_instructions", [])) if isinstance(item.get("step_by_step_instructions"), list) else None
                        or ""
                    )
                    content = content.strip()
                    if not content:
                        continue
                    
                    if len(content.split()) > MAX_CHUNK_SIZE:
                        chunks = smart_chunking(content, MAX_CHUNK_SIZE)
                        for i, chunk in enumerate(chunks):
                            chunk_entry = {
                                "title": item.get("title", "") + f" [Chunk {i+1}]",
                                "article_number": item.get("article_number", "") or item.get("id", "") or f"no-id-chunk{i}",
                                "content": chunk,
                                "raw": item,
                                "source_file": os.path.basename(f),
                                "is_chunk": True,
                                "parent_id": item.get("article_number", "") or item.get("id", "") or "no-id"
                            }
                            entries.append(chunk_entry)
                            texts.append(chunk)
                    else:
                        entry = {
                            "title": item.get("title", ""),
                            "article_number": item.get("article_number", "") or item.get("id", "") or "no-id",
                            "content": content,
                            "raw": item,
                            "source_file": os.path.basename(f),
                            "is_chunk": False
                        }
                        entries.append(entry)
                        texts.append(content)
                        
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
    
    print(f"✅ Loaded {len(entries)} KB entries (including chunks) from {len(files)} files.")
    return entries, texts

# Load knowledge base
knowledge_base, documents = load_kbs_with_smart_chunking(KB_GLOB)

# -------------------------
# Embeddings + FAISS
# -------------------------
print(f"🔁 Loading bi-encoder ({BI_MODEL})...")
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
    print(f"✅ Indexed {index.ntotal} vectors (dim={dim}).")
else:
    doc_embeddings = np.zeros((0,1), dtype="float32")
    doc_embeddings_norm = doc_embeddings
    index = None

# -------------------------
# BM25 setup
# -------------------------
if BM25_AVAILABLE and documents:
    tokenized_docs = [d.split() for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
else:
    bm25 = None

# -------------------------
# Cross-encoder
# -------------------------
if CROSS_AVAILABLE:
    try:
        print(f"🔁 Loading cross-encoder ({CROSS_MODEL})...")
        cross_encoder = CrossEncoder(CROSS_MODEL)
    except:
        cross_encoder = None
        CROSS_AVAILABLE = False
else:
    cross_encoder = None

# -------------------------
# Query Understanding & Analysis
# -------------------------
def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze query for intent, completeness, and context"""
    # Check if query is vague or incomplete
    word_count = len(query.split())
    has_question_words = any(word in query.lower() for word in ['how', 'what', 'why', 'when', 'where', 'which'])
    has_error_codes = bool(re.search(r'error\s+\d+|code\s+\d+', query.lower()))
    
    completeness_score = min(word_count / 10, 1.0)  # Scale based on word count
    
    # Check for specific technical patterns
    is_technical = any(term in query.lower() for term in [
        'error', 'issue', 'problem', 'not working', 'broken', 'fix', 'solve'
    ])
    
    return {
        "word_count": word_count,
        "has_question_words": has_question_words,
        "has_error_codes": has_error_codes,
        "completeness_score": completeness_score,
        "is_technical": is_technical,
        "is_vague": word_count < 4 and not has_error_codes,
        "needs_clarification": word_count < 6 and is_technical and not has_error_codes
    }

def needs_clarification(query: str, intent_analysis: Dict) -> bool:
    """Determine if clarification is needed"""
    if intent_analysis["is_vague"]:
        return True
    
    if intent_analysis["needs_clarification"]:
        return True
    
    # Check if this might be a follow-up to a previous technical issue
    if intent_analysis["is_technical"] and intent_analysis["completeness_score"] < 0.5:
        return True
    
    return False

def generate_clarification_questions(query: str, intent_analysis: Dict, session: Session) -> str:
    """Generate context-aware clarification questions"""
    base_questions = [
        "Could you provide more details about the issue?",
        "What specific error messages are you seeing?",
        "When did this problem start occurring?",
        "Have you tried any troubleshooting steps already?",
        "What is the exact system or application you're having trouble with?"
    ]
    
    # Context-aware questions based on conversation history
    context_questions = []
    if session.conversation_history:
        last_query = session.conversation_history[-1].get("query", "")
        if "network" in last_query.lower():
            context_questions.extend([
                "Are you connected to the VPN or corporate network?",
                "Can you ping the server or website you're trying to access?"
            ])
        elif "login" in last_query.lower():
            context_questions.extend([
                "Are you using the correct username and password?",
                "Have you tried resetting your password?"
            ])
        elif "email" in last_query.lower():
            context_questions.extend([
                "Are you able to send emails or just receive them?",
                "What email client are you using?"
            ])
    
    # Combine questions and select most relevant
    all_questions = base_questions + context_questions
    selected_questions = all_questions[:3]  # Limit to 3 questions
    
    clarification_prompt = f"""The user asked: "{query}"

This query seems incomplete for providing a precise solution. Please ask clarifying questions to get the necessary details.

Ask 2-3 specific questions that would help understand:
1. The exact nature of the problem
2. Any error messages or symptoms
3. What they've already tried

Response format:
- Start with a friendly acknowledgment
- Ask concise, specific questions
- Keep it professional and supportive"""

    try:
        response = call_gemini_raw(clarification_prompt, temperature=0.3, max_tokens=150)
        return response
    except:
        # Fallback response
        return f"I'd like to help you with \"{query}\". To provide the best assistance, could you please:\n\n" + "\n".join(f"• {q}" for q in selected_questions)

# -------------------------
# Enhanced Retrieval with Thresholds
# -------------------------
def improved_hybrid_retrieve(query: str, bi_top_k: int = BI_TOP_K, top_k: int = TOP_K) -> Tuple[List[Dict], Dict]:
    """Enhanced retrieval with relevance thresholds"""
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
        bm25_scores = bm25.get_scores(query.split())
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
        # Fallback to top results if threshold too strict
        filtered_results = fused_results[:top_k]
    
    final_indices = [idx for idx, score in filtered_results[:top_k]]

    # Prepare final candidates
    candidates = []
    for idx in final_indices:
        if idx >= len(knowledge_base):
            continue
            
        kb = knowledge_base[idx].copy()
        candidates.append({
            "idx": idx,
            "title": kb.get("title", ""),
            "article_number": kb.get("article_number", "no-id"),
            "content": kb.get("content", ""),
            "final_score": next((score for i, score in filtered_results if i == idx), 0.0)
        })

    # Cross-encoder re-ranking
    if cross_encoder is not None and candidates:
        pairs = [(query, c["content"][:1024] + " ...") for c in candidates]
        try:
            cross_scores = cross_encoder.predict(pairs)
            cross_scores = np.array(cross_scores, dtype=float)
            for i, score in enumerate(cross_scores):
                candidates[i]["cross_score"] = float(score)
                candidates[i]["final_score"] = candidates[i]["final_score"] * 0.7 + float(score) * 0.3
        except:
            for c in candidates:
                c["cross_score"] = 0.0

    # Sort by final score and apply cross-encoder threshold
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    candidates = [c for c in candidates if c.get("final_score", 0) >= RELEVANCE_THRESHOLD]
    
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
        "candidate_indices": [c["idx"] for c in candidates],
        "retrieval_time": retrieval_time,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "candidates_above_threshold": len(candidates)
    }
    
    return candidates, metadata

# -------------------------
# Utility Functions
# -------------------------
def softmax_conf(scores: np.ndarray) -> float:
    if len(scores)==0: return 0.0
    s = scores - np.max(scores)
    ex = np.exp(s)
    probs = ex / (np.sum(ex)+1e-12)
    return float(np.max(probs))

def is_exact_match(query: str, kb_entry: Dict) -> bool:
    q = query.strip().lower()
    title = (kb_entry.get("title") or "").lower()
    content = (kb_entry.get("content") or "").lower()
    if q in title or q in content: return True
    if len(q.split()) <= 4:
        q_tokens = set(q.split())
        content_tokens = set(content.split())
        return q_tokens.issubset(content_tokens)
    return False

def normalize_query(query: str) -> str:
    return ' '.join(query.strip().lower().split())

def get_query_hash(query: str) -> str:
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

# -------------------------
# LLM Integration
# -------------------------
def call_gemini_raw(prompt: str, model: str = DEFAULT_GEMINI_MODEL, temperature: float = 0.0, max_tokens: int = 800) -> str:
    if not GEMINI_API_KEY or not GENAI_AVAILABLE:
        return "[LLM disabled] GEMINI_API_KEY missing or SDK not available."
    try:
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
        if hasattr(resp, "text") and isinstance(resp.text, str): 
            return resp.text
        return str(resp)
    except Exception as e:
        return f"[LLM call failed: {str(e)}]"

# -------------------------
# Enhanced Prompt Templates
# -------------------------
PROMPT_KB_TEMPLATE = """**Role**: L1 Support Co-Pilot for new analysts
**Task**: Answer the user's question using ONLY the provided Knowledge Base content. Extract and present the information directly.

**Constraints**: 
- Use ONLY the KB content provided below
- Do NOT tell users to "refer to KB article X" - provide the actual information
- Present the solution in clear, natural language
- Focus on actionable steps
- Keep explanations beginner-friendly

**Response Format**:
1. **Brief Summary**: One sentence overview of the solution
2. **Step-by-Step Resolution**: Numbered exact steps from the KB content
3. **Expected Outcome**: What should happen after following the steps
4. **Escalation Path**: When and how to escalate if the solution doesn't work

**User Query**: {query}

**Relevant KB Content**:
{kb_block}

**Instructions**: Extract the exact procedure from the KB content and present it in a clear, natural way. Do not mention KB article numbers or tell users to "refer to" anything - provide the actual information directly."""

CONTEXT_AWARE_TEMPLATE = """**Role**: L1 Support Co-Pilot with Conversation Memory
**Task**: Continue from previous discussion to answer the follow-up question

**Previous Interaction**:
- User asked: {previous_query}
- You answered: {previous_answer}

**Current Question**: {query}

**Instructions**:
1. Build upon your previous answer - don't repeat it entirely
2. Provide additional details or clarification specifically requested
3. If the question is about a specific step from previous answer, elaborate on that step
4. Maintain the same mentoring tone
5. Keep it concise but complete

**Important**: Only reference the previous answer, don't introduce new unrelated information."""

PROMPT_FALLBACK_TEMPLATE = """**Role**: L1 Support Co-Pilot for new analysts
**Task**: Provide helpful, mentoring guidance when specific KB content isn't available

**Context**: 
{conversation_context}

**Current Query**: {query}

**Situation**: No specific knowledge base content was found for this query, or the available content wasn't sufficient.

**Instructions**:
1. Provide general best practices and troubleshooting guidance
2. Offer mentoring advice appropriate for L1 analysts
3. Suggest common steps they might try
4. Clearly state that this is general advice, not from a specific KB article
5. If appropriate, guide them on what information would help provide a better answer
6. Maintain a supportive, mentoring tone

**Important**: Be honest that this isn't from a specific KB article, but still provide helpful guidance."""

# -------------------------
# Response Generation
# -------------------------
def build_conversation_context(session: Session, limit: int = 3) -> str:
    """Build conversation context for the LLM"""
    if not session.conversation_history:
        return "No previous conversation context."
    
    context_lines = []
    for turn in session.conversation_history[-limit:]:
        context_lines.append(f"User: {turn.get('query', '')}")
        context_lines.append(f"Assistant: {turn.get('answer', '')[:200]}...")
    
    return "\n".join(context_lines)

def build_kb_block(candidates: List[Dict]) -> str:
    """Build efficient KB context block"""
    if not candidates:
        return "No relevant knowledge base content found."
    
    sorted_candidates = sorted(
        candidates, 
        key=lambda x: x.get('final_score', x.get('cross_score', 0)), 
        reverse=True
    )
    
    blocks = []
    max_blocks = 3
    
    for i, c in enumerate(sorted_candidates[:max_blocks]):
        content = c.get("content", "").strip()
        if not content:
            continue
            
        if len(content) > 800:
            trunc_point = content[:800].rfind('. ')
            if trunc_point > 400:
                content = content[:trunc_point + 1] + " [content truncated]"
            else:
                content = content[:600] + " [content truncated]"
        
        block_info = f"[Source: {c.get('article_number', 'KB')}]"
        if c.get('title'):
            block_info += f" {c['title']}"
            
        blocks.append(f"{block_info}\n{content}")
    
    return "\n\n---\n\n".join(blocks)

def generate_answer(query: str, candidates: List[Dict], metadata: Dict[str, Any], 
                   session: Session, previous_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Generate answer with conversation context"""
    start_time = time.time()
    
    used_kb = metadata.get("used_kb", False)
    exact_match = metadata.get("exact_match", False)
    confidence = float(metadata.get("confidence", 0.0))

    # Build conversation context
    conversation_context = build_conversation_context(session)

    if previous_context:
        # Context-aware follow-up
        prompt = CONTEXT_AWARE_TEMPLATE.format(
            query=query,
            previous_answer=previous_context.get("answer", "")[:1000],
            previous_query=previous_context.get("query", "")
        )
        resp_text = call_gemini_raw(prompt)
        used_kb = previous_context.get("used_kb", False)
    elif used_kb and candidates:
        # KB-based answer
        kb_block = build_kb_block(candidates)
        
        # Log KB matches
        print(f"\n🔹 KB matches for query: {query}")
        for c in candidates[:3]:
            kb = knowledge_base[c["idx"]]
            print(f"   [KB-ID: {kb.get('article_number','no-id')}] "
                  f"Score: {c.get('final_score', 0):.3f} - "
                  f"{kb.get('title','(no title)')}")

        prompt = PROMPT_KB_TEMPLATE.format(
            query=query, 
            kb_block=kb_block,
            conversation_context=conversation_context
        )
        resp_text = call_gemini_raw(prompt)
        
        # Fallback if KB content doesn't help
        if any(phrase in resp_text.lower() for phrase in ["i don't know", "not covered", "no information"]):
            prompt = PROMPT_FALLBACK_TEMPLATE.format(
                query=query,
                conversation_context=conversation_context
            )
            resp_text = call_gemini_raw(prompt)
            used_kb = False
    else:
        # General mentor answer
        prompt = PROMPT_FALLBACK_TEMPLATE.format(
            query=query,
            conversation_context=conversation_context
        )
        resp_text = call_gemini_raw(prompt)

    generation_time = time.time() - start_time

    # Store in DB
    try:
        con = get_db_connection()
        cur = con.cursor()
        top_ids = ",".join(str(c["idx"]) for c in candidates[:TOP_K]) if candidates else ""
        cur.execute("""
            INSERT INTO history (session_id, query, answer, confidence, used_kb, exact_match, top_k_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session.session_id, query, resp_text, confidence, int(used_kb), int(exact_match), top_ids))
        con.commit()
        con.close()
    except Exception as e:
        print(f"❌ Database error: {e}")

    return {
        "answer": resp_text,
        "confidence": confidence,
        "used_kb": used_kb,
        "exact_match": exact_match,
        "candidates": candidates,
        "generation_time": generation_time,
        "used_previous_context": previous_context is not None
    }

# -------------------------
# Main Query Function with Enhanced Capabilities
# -------------------------
def query_rag(query: str, session_id: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
    """Enhanced query function with all new capabilities"""
    total_start = time.time()
    
    # Get or create session
    session = get_session(session_id, user_id)
    
    # Analyze query intent
    intent_analysis = analyze_query_intent(query)
    
    # Check if clarification is needed
    if needs_clarification(query, intent_analysis) and not session.pending_clarification:
        clarification_response = generate_clarification_questions(query, intent_analysis, session)
        
        # Store pending clarification
        session.pending_clarification = {
            "original_query": query,
            "clarification_response": clarification_response,
            "timestamp": datetime.now()
        }
        
        save_session_to_db(session)
        
        return {
            "answer": clarification_response,
            "confidence": 0.3,
            "used_kb": False,
            "exact_match": False,
            "clarification_asked": True,
            "session_id": session.session_id,
            "needs_follow_up": True
        }
    
    # Handle follow-up to clarification
    if session.pending_clarification:
        # This is a follow-up to a clarification request
        enhanced_query = f"{session.pending_clarification['original_query']} Additional details: {query}"
        session.pending_clarification = None
    else:
        enhanced_query = query
    
    # Retrieve information
    candidates, metadata = improved_hybrid_retrieve(enhanced_query)
    
    # Check conversation history for context
    previous_context = None
    if session.conversation_history:
        previous_context = session.conversation_history[-1]
    
    # Generate answer
    result = generate_answer(enhanced_query, candidates, metadata, session, previous_context)
    
    # Update session history
    session.conversation_history.append({
        "query": query,
        "answer": result["answer"],
        "timestamp": datetime.now().isoformat(),
        "used_kb": result["used_kb"],
        "confidence": result["confidence"]
    })
    
    # Keep history manageable
    if len(session.conversation_history) > 10:
        session.conversation_history.pop(0)
    
    save_session_to_db(session)
    
    # Add timing information
    result.update({
        "session_id": session.session_id,
        "retrieval_time": metadata.get("retrieval_time", 0),
        "total_time": time.time() - total_start,
        "clarification_asked": False,
        "intent_analysis": intent_analysis
    })
    
    return result

# -------------------------
# Feedback System
# -------------------------
def submit_feedback(session_id: str, rating: int, comment: str = "") -> bool:
    """Submit user feedback for a session"""
    try:
        # Get the last interaction from this session
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("""
            SELECT query, answer FROM history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (session_id,))
        row = cur.fetchone()
        
        if row:
            query, answer = row
            cur.execute("""
                INSERT INTO feedback (session_id, query, answer, rating, comment)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, query, answer, rating, comment))
            
            # Also update the history record with feedback
            cur.execute("""
                UPDATE history SET user_feedback = ?, feedback_comment = ?
                WHERE session_id = ? AND timestamp = (
                    SELECT MAX(timestamp) FROM history WHERE session_id = ?
                )
            """, (rating, comment, session_id, session_id))
            
            con.commit()
        
        con.close()
        return True
    except Exception as e:
        print(f"❌ Error submitting feedback: {e}")
        return False

def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        cur.execute("SELECT COUNT(*), AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        total, avg_rating = cur.fetchone()
        
        cur.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
        rating_counts = {str(rating): count for rating, count in cur.fetchall()}
        
        con.close()
        
        return {
            "total_feedback": total,
            "average_rating": round(avg_rating, 2) if avg_rating else 0,
            "rating_distribution": rating_counts
        }
    except Exception as e:
        print(f"❌ Error getting feedback stats: {e}")
        return {}
    
# -------------------------
# Enhanced History functions 
# -------------------------
def get_last_turns(limit: int = 3) -> List[Dict[str, Any]]:
    """Get the last conversation turns from database (across all sessions)"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("""
            SELECT query, answer, confidence, used_kb, exact_match, clarification_asked, timestamp 
            FROM history ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        con.close()
        
        return [{
            "query": r[0],
            "answer": r[1],
            "confidence": float(r[2]),
            "used_kb": bool(r[3]),
            "exact_match": bool(r[4]),
            "clarification_asked": bool(r[5]),
            "timestamp": r[6]
        } for r in rows]
    except Exception as e:
        print(f"❌ Error getting last turns: {e}")
        return []

def get_conversation_history(session_id: str, limit: int = 10) -> List[Dict]:
    """Get conversation history for a specific session"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("""
            SELECT query, answer, confidence, used_kb, exact_match, clarification_asked, timestamp 
            FROM history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit))
        rows = cur.fetchall()
        con.close()
        
        return [{
            "query": r[0],
            "answer": r[1],
            "confidence": float(r[2]),
            "used_kb": bool(r[3]),
            "exact_match": bool(r[4]),
            "clarification_asked": bool(r[5]),
            "timestamp": r[6]
        } for r in rows]
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        return []

def get_session_overview(limit: int = 20) -> List[Dict]:
    """Get overview of recent sessions with stats"""
    try:
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("""
            SELECT 
                session_id,
                COUNT(*) as total_messages,
                AVG(confidence) as avg_confidence,
                SUM(used_kb) as kb_usage_count,
                MAX(timestamp) as last_activity
            FROM history 
            GROUP BY session_id 
            ORDER BY last_activity DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        con.close()
        
        return [{
            "session_id": r[0],
            "total_messages": r[1],
            "avg_confidence": round(float(r[2]), 3) if r[2] else 0,
            "kb_usage_rate": round(float(r[3]) / float(r[1]), 3) if r[1] > 0 else 0,
            "last_activity": r[4]
        } for r in rows]
    except Exception as e:
        print(f"❌ Error getting session overview: {e}")
        return []

# -------------------------
# Performance Monitoring
# -------------------------
# Continue from the last line...

def get_performance_stats(limit: int = 100) -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    if not performance_log:
        return {}
    
    recent_logs = performance_log[-limit:] if len(performance_log) > limit else performance_log
    
    response_times = [m.query_time for m in recent_logs]
    retrieval_times = [m.retrieval_time for m in recent_logs if m.retrieval_time > 0.001]
    generation_times = [m.generation_time for m in recent_logs if m.generation_time > 0.001]
    confidences = [m.confidence for m in recent_logs]
    
    cache_hits = sum(1 for m in recent_logs if m.cache_hit)
    kb_usage = sum(1 for m in recent_logs if m.kb_used)
    exact_matches = sum(1 for m in recent_logs if m.exact_match)
    context_usage = sum(1 for m in recent_logs if m.used_context)
    clarifications = sum(1 for m in recent_logs if m.clarification_asked)
    
    return {
        "total_queries": len(recent_logs),
        "cache_hit_rate": cache_hits / len(recent_logs) if recent_logs else 0,
        "kb_usage_rate": kb_usage / len(recent_logs) if recent_logs else 0,
        "exact_match_rate": exact_matches / len(recent_logs) if recent_logs else 0,
        "context_usage_rate": context_usage / len(recent_logs) if recent_logs else 0,
        "clarification_rate": clarifications / len(recent_logs) if recent_logs else 0,
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
        "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "p95_response_time": np.percentile(response_times, 95) if response_times else 0,
        "max_response_time": max(response_times) if response_times else 0
    }

def clear_performance_log():
    """Clear performance monitoring logs"""
    global performance_log
    performance_log = []

def get_recent_queries(limit: int = 20) -> List[Dict]:
    """Get recent query performance data"""
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("""
        SELECT query, answer, confidence, used_kb, exact_match, clarification_asked, timestamp 
        FROM history ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    
    return [{
        "query": r[0], 
        "answer": r[1][:100] + "..." if len(r[1]) > 100 else r[1],
        "confidence": float(r[2]), 
        "used_kb": bool(r[3]), 
        "exact_match": bool(r[4]), 
        "clarification_asked": bool(r[5]),
        "timestamp": r[6]
    } for r in rows]

def get_session_history(session_id: str) -> List[Dict]:
    """Get complete history for a session"""
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("""
        SELECT query, answer, confidence, used_kb, exact_match, clarification_asked, timestamp 
        FROM history WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,))
    rows = cur.fetchall()
    con.close()
    
    return [{
        "query": r[0], 
        "answer": r[1],
        "confidence": float(r[2]), 
        "used_kb": bool(r[3]), 
        "exact_match": bool(r[4]), 
        "clarification_asked": bool(r[5]),
        "timestamp": r[6]
    } for r in rows]

# -------------------------
# Diagnostic Functions
# -------------------------
def diagnose_retrieval(query: str):
    """Comprehensive retrieval diagnosis"""
    print(f"🧪 DIAGNOSING: '{query}'")
    print("=" * 60)
    
    # Check exact matches
    check_exact_matches(query)
    print()
    
    # Test retrieval (bypass cache)
    query_hash = get_query_hash(query)
    if query_hash in query_cache:
        del query_cache[query_hash]
    
    candidates, metadata = improved_hybrid_retrieve(query)
    print(f"Retrieval confidence: {metadata['confidence']:.3f}")
    print(f"Used KB: {metadata['used_kb']}")
    print(f"Exact match: {metadata['exact_match']}")
    print(f"Candidates above threshold: {metadata['candidates_above_threshold']}")
    print()
    
    # Show top candidates
    debug_retrieval(query, candidates)
    
    # Test cross-encoder if available
    if candidates and CROSS_AVAILABLE and cross_encoder:
        test_cross_encoder(query, candidates)
    
    print("=" * 60)
    return candidates, metadata

def check_exact_matches(query: str):
    """Check for exact matches in knowledge base"""
    print(f"🔎 Checking exact matches for: '{query}'")
    exact_matches = []
    for i, kb_item in enumerate(knowledge_base):
        if is_exact_match(query, kb_item):
            exact_matches.append((i, kb_item))
    
    if exact_matches:
        print(f"✅ Found {len(exact_matches)} exact match(es):")
        for idx, (i, kb_item) in enumerate(exact_matches[:3]):
            print(f"   {idx+1}. Index {i}: {kb_item.get('title', 'No title')}")
            print(f"      Content: {kb_item.get('content', '')[:100]}...")
    else:
        print("❌ No exact matches found")
    return exact_matches

def debug_retrieval(query: str, candidates: List[Dict]):
    """Debug retrieval results"""
    print(f"🔍 Top retrieval candidates for: '{query}'")
    if not candidates:
        print("❌ No candidates retrieved")
        return
    
    for i, candidate in enumerate(candidates[:5]):
        kb_item = knowledge_base[candidate["idx"]]
        print(f"{i+1}. Score: {candidate.get('final_score', 0):.3f}")
        if 'cross_score' in candidate:
            print(f"   Cross-score: {candidate['cross_score']:.3f}")
        print(f"   Title: {kb_item.get('title', 'No title')}")
        print(f"   Article #: {kb_item.get('article_number', 'N/A')}")
        print(f"   Source: {kb_item.get('source_file', 'Unknown')}")
        print(f"   Content: {kb_item.get('content', '')[:150]}...")
        print("---")

def test_cross_encoder(query: str, candidates: List[Dict]):
    """Test cross-encoder functionality"""
    print("🧠 Testing cross-encoder...")
    try:
        test_text = candidates[0].get("content", "")[:512]
        test_score = cross_encoder.predict([(query, test_text)])
        print(f"   Cross-encoder test score: {test_score[0]:.3f}")
        print(f"   Status: ✅ Working")
    except Exception as e:
        print(f"   ❌ Cross-encoder error: {e}")

def validate_knowledge_base():
    """Validate KB content and structure"""
    print("📋 Knowledge Base Validation")
    print("=" * 40)
    
    print(f"Total KB entries: {len(knowledge_base)}")
    print(f"Total documents: {len(documents)}")
    print(f"FAISS index size: {index.ntotal if index else 0}")
    print()
    
    # Check first 5 items
    empty_count = 0
    for i, item in enumerate(knowledge_base[:5]):
        content = item.get('content', '')
        is_empty = not content.strip()
        if is_empty:
            empty_count += 1
        
        print(f"{i+1}. Words: {len(content.split())}, Chars: {len(content)}")
        print(f"   Title: {item.get('title', 'None')}")
        print(f"   Empty: {is_empty}")
        print(f"   Preview: {content[:80]}{'...' if len(content) > 80 else ''}")
        print("---")
    
    if empty_count > 0:
        print(f"⚠️  Warning: {empty_count} empty content items in sample")

def test_embedding_quality():
    """Test if embeddings are working properly"""
    print("🔤 Testing embedding quality...")
    test_queries = ["password reset", "network issue", "software installation"]
    
    for query in test_queries:
        q_emb = bi_encoder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, 3)
        print(f"Query: '{query}' -> Top scores: {D[0]}")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Test the enhanced RAG system
    print("🚀 Testing Enhanced RAG System with Conversational Capabilities")
    print("=" * 70)
    
    # Create a test session
    session = create_session("test_user")
    
    # Test 1: Vague query that should trigger clarification
    print("\n1. Testing vague query (should ask for clarification):")
    test_query_1 = "Login issue"
    result_1 = query_rag(test_query_1, session.session_id)
    print(f"Query: '{test_query_1}'")
    print(f"Response: {result_1['answer'][:200]}...")
    print(f"Clarification asked: {result_1.get('clarification_asked', False)}")
    print(f"Session ID: {result_1['session_id']}")
    
    # Test 2: Follow-up with details
    print("\n2. Testing follow-up with details:")
    test_query_2 = "I'm getting error 401 when trying to access the portal"
    result_2 = query_rag(test_query_2, session.session_id)
    print(f"Query: '{test_query_2}'")
    print(f"Response: {result_2['answer'][:200]}...")
    print(f"Used KB: {result_2['used_kb']}")
    print(f"Confidence: {result_2['confidence']:.3f}")
    
    # Test 3: Show session history
    print("\n3. Session History:")
    history = get_session_history(session.session_id)
    for i, turn in enumerate(history):
        print(f"  {i+1}. {turn['query']} -> {turn['answer'][:50]}...")
    
    # Test 4: Performance stats
    print("\n4. Performance Statistics:")
    stats = get_performance_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    # Test 5: Feedback submission
    print("\n5. Testing feedback system:")
    feedback_result = submit_feedback(session.session_id, 5, "Very helpful response!")
    print(f"Feedback submitted: {feedback_result}")
    
    feedback_stats = get_feedback_stats()
    print(f"Feedback stats: {feedback_stats}")
    
    print("\n✅ Enhanced RAG system test completed successfully!")
