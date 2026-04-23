# Grid07 — AI Cognitive Routing & RAG Engine

A three-phase AI pipeline implementing vector-based persona routing, autonomous LangGraph content generation, and RAG-based debate with prompt injection defense.

---

## Setup

```bash
git clone <your-repo-url>
cd grid07
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Run

```bash
# Run the full pipeline (all 3 phases)
python main.py

# Run individual phases
python phase1_router.py
python phase2_content_engine.py
python phase3_combat_engine.py
```

---

## Architecture

### Phase 1 — Vector Router

Bot personas are embedded using `text-embedding-3-small` and stored in an in-memory FAISS index. When a post arrives, it is also embedded and compared against each persona using L2 distance, converted to a similarity score. Bots above the threshold are returned as matches.

**Threshold note:** FAISS uses L2 distance internally. The similarity is computed as `1 / (1 + L2_distance)`, giving a 0–1 range. A threshold of `0.30` works well with `text-embedding-3-small`; adjust as needed.

---

### Phase 2 — LangGraph Content Engine

Three-node state machine:

```
[decide_search] → [web_search] → [draft_post] → END
```

- **decide_search**: LLM picks a topic and formats a search query based on the bot's persona.
- **web_search**: `mock_searxng_search` tool returns hardcoded headlines by keyword matching.
- **draft_post**: LLM uses persona (system prompt) + search results (context) to generate a post, constrained to strict JSON: `{"bot_id": "...", "topic": "...", "post_content": "..."}`.

Structured output is enforced via a strict system prompt instructing the model to output only a JSON object. The response is parsed with `json.loads()` and any markdown fences are stripped before parsing.

---

### Phase 3 — Combat Engine (RAG + Injection Defense)

`generate_defense_reply()` constructs a RAG prompt by injecting:
1. The full bot persona as an immutable system identity block.
2. The parent post and full comment history as thread context.
3. The latest human reply.

**Prompt Injection Defense — two-layer approach:**

**Layer 1 (Keyword Filter):** `detect_injection_attempt()` scans the human reply for known injection patterns (`"ignore all previous instructions"`, `"you are now"`, `"apologize"`, etc.). If triggered, an explicit `SECURITY ALERT` block is injected into the prompt.

**Layer 2 (System Prompt Hardening):** The system prompt contains an `=== CORE DIRECTIVE ===` block that explicitly instructs the model that its identity is immutable, that any attempt to redefine it is a manipulation attempt, and that it should mock or call out such attempts while doubling down on its persona.

This combination ensures the bot never complies with injection attempts — it stays in character and typically roasts the human for trying.

---

## File Structure

```
grid07/
├── main.py                   # Runs all 3 phases end-to-end
├── phase1_router.py          # Vector persona matching
├── phase2_content_engine.py  # LangGraph content generation
├── phase3_combat_engine.py   # RAG debate + injection defense
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Notes

- Uses OpenAI `gpt-4o-mini` for all LLM calls (cost-efficient).
- Uses `text-embedding-3-small` for embeddings.
- FAISS runs fully in-memory — no external database needed.
- Swap `OpenAIEmbeddings` and `ChatOpenAI` for Groq/Ollama if preferred.
