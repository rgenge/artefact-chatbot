# Empório da Música — Gemini agent

Python prototype of the customer-service agent asked for in the challenge. The
architecture follows the TwinTweaker process:

1. Catalog, promotions and orders are queried structurally from the CSVs.
2. The PDF manual is split into 1,400-character chunks with 180 of overlap.
3. The PDF uses hybrid retrieval: deterministic keyword ranking + Gemini
   embeddings (`gemini-embedding-001`, 768 dimensions).
4. The retrieved context goes to `gemini-3.1-flash`, which writes short, warm
   open-ended answers.
5. Structured catalog lists stay deterministic: the model may not summarise,
   truncate or invent price/stock rows.
6. With no key or on a network failure, the local fallback keeps working without
   inventing price or stock.

The agent replies to customers in Brazilian Portuguese — the store is in Campo
Grande/MS and that is its persona. This documentation is in English, as required
for the submission.

## Setup

~~~bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
~~~

Fill in `.env`:

~~~env
GOOGLE_GENERATIVE_AI_API_KEY=your_token
GEMINI_MODEL=gemini-3.1-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
~~~

The key never lives in the code. `.env` is already in `.gitignore`.

## Usage

With Gemini enabled:

~~~bash
python app.py
~~~

Single query:

~~~bash
python app.py --message "Quanto custa o Takamine GD20?"
~~~

Order lookup with identification:

~~~bash
python app.py --customer-id 2 --message "Qual o status do pedido 8?"
~~~

Browser UI at `http://127.0.0.1:8000`:

~~~bash
python app.py --web
~~~

Offline mode, no external calls:

~~~bash
python app.py --no-llm
python -m unittest discover -s tests -v
~~~

Reproducible offline evaluation of real conversations:

~~~bash
python tests_ai/run_evaluation.py
~~~

The report is written to `tests_ai/report.md` and the detailed result to
`tests_ai/results.json`.

## Architecture

The flow separates two sources of truth.

**Catalog — structured, never RAG.** Products, categories, promotions,
customers and orders are loaded from the CSVs and queried with filters,
normalisation and deterministic joins. Price, stock, discount and order status
are never guessed by semantic similarity. The agent corrects common brand typos,
reports the total when it shows a five-item preview, and keeps context across
follow-ups like "só esses?" and "E da Tagima?".

**Policy manual — hybrid RAG.** Keywords preserve exact terms, codes and rules;
embeddings retrieve semantically similar phrasings. Price/model/policy questions
prioritise lexical evidence, while narrative questions may lean on semantic
similarity. Gemini only writes over the retrieved context, and a deterministic
guard rejects answers that introduce monetary values absent from that context.

**Prompt and history.** Recent history is replayed to Gemini as user/model
turns. The prompt constrains scope, tone, language and length, and forbids
inventing data. Accessory requests are redirected per the manual.

**Human handoff.** A narrow set of deterministic triggers — late delivery,
non-delivery, damage on arrival, an explicit request for a person, a mention of
Procon — routes the message to a handoff instead of an answer. The web UI shows
the trigger list and lets the customer switch escalation off.

The web UI (`python app.py --web`) uses exactly the same `StoreAgent`: every
question is routed to catalog, order, policy or handoff without duplicating
logic.

See [examples/conversations.md](examples/conversations.md) for the required
scenarios.

## Why these choices

- **Hybrid instead of pure RAG for the catalog.** Price, stock and status need
  exact filters and joins; semantic similarity gets numbers wrong. The CSVs
  become deterministic queries and only the policy PDF goes to RAG.
- **Gemini 3.1 Flash, the cheap model.** Here the LLM only writes over an
  already-retrieved context — the hard work belongs to the retriever. An
  expensive model would not improve grounding and would raise the cost of every
  turn.
- **Deterministic triggers for human handoff.** A late order, damaged goods or
  an explicit request for a person have no correct answer in the data; any
  generated text would be an empty promise. The rule escalates immediately, and
  the trigger list is visible in the UI.

## Use of code assistants

Claude Code (Claude Opus, in the VS Code extension) was used throughout, as a
pair programmer rather than a code generator: I described the intent and the
constraint, reviewed each diff, and kept the architectural decisions — the
structured/RAG split, the cheap model, the narrow handoff triggers — my own. It
was most useful for mechanical breadth: sweeping the retrieval helpers, keeping
the regex sets and their readable UI labels in sync, and running the unit tests
and the conversation evaluation after each change so regressions surfaced
immediately.

## Known limitations

The database is a local snapshot and there is no real customer authentication.
The embedding index is rebuilt at process start instead of persisting to
pgvector/Supabase as in TwinTweaker. The human handoff already detects and
records the case, but dispatch to the support queue does not exist yet.

With more time I would persist chunks and embeddings, add incremental updates,
observability, and continuous recall/grounding evaluation.
