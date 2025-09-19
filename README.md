📑 GenAI Powered L1 Analyst Co-Pilot
Purpose and Rationale
The goal of this project is to design and build an intelligent, GenAI-powered co-pilot solution for Level 1 (L1) analysts in production support teams. This co-pilot provides instant, context-aware assistance by querying a structured, categorized Knowledge Base (KB) of recurring production issues, troubleshooting steps, and best practices.
The solution aims to:
⚡ Accelerate issue resolution by delivering targeted guidance instantly.
📉 Reduce escalations to higher support levels by empowering L1 analysts.
🧠 Preserve and leverage institutional knowledge efficiently through KB-driven retrieval.
🚀 Enable faster onboarding and continuous upskilling of analysts.
💬 Provide a conversational interface that feels natural and context-aware.

This approach ensures clarity of needs, edge-case handling, and avoids “premature coding,” resulting in a robust, production-aligned MVP that balances accuracy, reliability, and cost-efficiency.
Current Problems in L1 Analyst Support
High Dependency on Senior Engineers
L1 analysts escalate many cases due to lack of quick, trusted answers.
Information Overload
Knowledge is scattered across wikis, documents, and old chat transcripts, leading to inefficiency.
Inconsistent Response Quality
Analysts may misinterpret documents, leading to errors or delays.
Slow Onboarding
New analysts take weeks to become effective because institutional knowledge is not easily accessible.
Generic GenAI Shortcomings
Pure LLM-based chatbots hallucinate, lack grounding in KB, and can’t always ensure trustworthy responses.
Our Approach
We combined traditional IR (Information Retrieval) methods + modern GenAI models to balance precision, recall, cost-efficiency, and explainability.

Key pillars:
📚 Structured Knowledge Base (KB): Curated JSON-based KB of production issues & resolutions.
🔍 Hybrid Retrieval: Bi-encoder semantic search + BM25 lexical matching.
⚖ Re-ranking Layer: Lightweight Cross-Encoder applied only to top candidates for accuracy.
🧩 Chunking Strategy: Overlap-aware splitting of documents to preserve context.
🎛 Confidence-Based Decisioning: Dynamic thresholds + disclaimers (Green/Yellow/Red) to reflect reliability.
📝 Feedback Loop: Analyst feedback stored in SQLite DB to continuously refine model relevance.
📊 Metrics Dashboard: Tracks usage, fallback counts, top queries, confidence score trends.


Technical Workflow
User Query Input
Analyst enters a natural language question in the chat UI.
Hybrid Candidate Retrieval
Bi-encoder embeddings: Dense semantic search for meaning-based matches.
BM25: Lexical match for keyword-driven precision.
Union of results: Ensures coverage of both “keyword-heavy” and “semantic” queries.
Candidate Re-Ranking (Cross-Encoder)
Top-N candidates passed through a lightweight cross-encoder.
Produces pairwise relevance scores (query + chunk).
Ensures correct ordering even when keywords overlap but meaning differs.

Confidence Calculation
Softmax-normalized cross-scores → Probability distribution.
Exact matches boosted to high confidence (≥0.95).
Otherwise thresholds applied (green/yellow/red disclaimer).

Answer Generation
KB-aligned chunk surfaced directly (trusted).
If confidence low → fallback disclaimer prompts human verification.

Feedback Collection
Analyst can mark answer as helpful/unhelpful.
Feedback stored in SQLite (persistent, not cache).

Metrics & Monitoring
Tracks fallback %, average confidence, top queries.
Identifies weak KB areas to refine content.

Main Backend Engine Concepts
Bi-Encoder (SentenceTransformers): Efficient retrieval, cosine similarity, scalable.
BM25: Classical keyword retrieval, ensures exact terms are not missed.
Cross-Encoder (Optional in MVP, Full in Budgeted): Improves precision by jointly encoding query + document. Lightweight model for free tier; scalable upgrade in AWS.
Confidence & Disclaimer Mechanism: Dynamic reliability classification prevents blind trust in low-confidence answers.
Chunking with Overlap: Ensures no context boundary loss. Reduces false negatives in retrieval.
Why Our Product is Efficient vs. Generic GenAI Solutions
Grounded in KB → No hallucinations; always aligned to real organizational data.
Hybrid Retrieval → Covers both exact matches and semantic variations.
Explainability → Confidence + disclaimers give transparency in decisioning.
Lightweight MVP → Free-tier friendly (bi + BM25 + optional lightweight cross).
Feedback Loop → Self-improving model with real analyst input.
Scalable Design → Easily enhanced with:
Full cross-encoder on AWS.
RAG orchestration in agentic setups.
Multi-source KB integration.
