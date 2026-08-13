# Empório da Música — Gemini customer-service agent

Python prototype of the customer-service agent for the Artefact AI Engineer
challenge. Empório da Música is a fictional musical-instrument store in Campo
Grande/MS whose team is overloaded with recurring questions: opening hours,
order status, price and availability.

The retrieval split reuses the approach from TwinTweaker, a previous RAG project
of mine: tabular facts are queried structurally and never embedded, while free
text goes through hybrid retrieval.

1. Catalog, promotions and orders are queried structurally from the CSVs.
2. The PDF manual is split into 1,400-character chunks with 180 of overlap.
3. The PDF uses hybrid retrieval: deterministic keyword ranking + Gemini
   embeddings (`gemini-embedding-001`, 768 dimensions).
4. The retrieved context goes to `gemini-3.1-flash-lite`, which writes short,
   warm open-ended answers.
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
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
~~~

The store data (`data/`) is committed, so the project runs straight after
cloning. A Gemini key is optional — without one the agent falls back to
deterministic retrieval and still answers correctly.

To enable Gemini, copy the example env file (`copy` on Windows, `cp` elsewhere)
and fill in your key:

~~~env
GOOGLE_GENERATIVE_AI_API_KEY=your_token
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
~~~

The key never lives in the code. `.env` is already in `.gitignore`.

## Usage

~~~bash
python app.py                                              # interactive CLI
python app.py --message "Quanto custa o Takamine GD20?"    # single query
python app.py --customer-id 2 --message "Qual o status do pedido 8?"
python app.py --web                                        # UI at 127.0.0.1:8000
python app.py --no-llm                                     # no external calls
~~~

Tests and evaluation:

~~~bash
python -m unittest discover -s tests -v    # unit tests
python tests_ai/run_evaluation.py          # multi-turn conversation evaluation
python tests_ai/run_evaluation.py --live   # same assertions, through Gemini
~~~

The evaluation writes `tests_ai/report.md` and `tests_ai/results.json`. It
checks every answer against the CSV rows and the policy PDF text, so a
regression in retrieval fails the run rather than producing plausible prose.

## Technical decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | None — standard library, plus `pypdf` to read the manual | The whole problem is routing, retrieval and grounding. An agent framework would add a dependency surface and hide exactly the logic being assessed. Every decision stays inspectable and testable. |
| Model / provider | Gemini 3.1 Flash Lite via REST | The LLM only writes over already-retrieved context — the hard work belongs to the retriever. A more expensive model would not improve grounding and would raise the cost of every turn. |
| Retrieval | Hybrid: structured CSV queries + keyword/embedding RAG on the PDF | Price, stock and status need exact filters and joins; semantic similarity gets numbers wrong. Only free-text policy goes to RAG. |
| Interface | CLI plus a dependency-free browser UI | The CLI keeps the agent scriptable and testable; the UI shows the same `StoreAgent` in a realistic setting. Both call one code path, so neither can drift. |
| History | In memory, per session, replayed as user/model turns | Enough for multi-turn context ("só esses?", "E da Tagima?") without a database the prototype does not need. |
| Data treatment | Normalisation, accent/typo tolerance, deterministic joins | Real customers write "yahama" and "violao". Matching happens on normalised text while answers quote the exact catalog row. |

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

**Prompt strategy.** Recent history is replayed to Gemini as user/model turns.
The system prompt fixes the persona, constrains scope, tone, language and
length, and forbids inventing price, stock, promotion, deadline or rule. When
information is missing the agent must say it needs to confirm with the team
rather than fill the gap.

**Refusals stay deterministic.** The model is only asked to rewrite when there
is retrieved evidence to rewrite from. With no grounding the answer is a
deterministic refusal — "identify yourself before I show order data", "no rule
found for that" — and a free rewrite would drop the guarantee it carries.
Prompt-injection attempts are answered without naming what is protected.

**Exchange.** A return/exchange request keeps its own context across turns, so
the customer can name the product and purchase date afterwards and still get the
policy window applied, without loading customer or order tables.

**Checkout.** Buying is a separate flow from an order lookup: it keeps the
chosen product, quantity and delivery address across turns, and loads no
customer or order tables. It deliberately does not create an order — the agent
says so explicitly rather than implying a purchase was registered.

**Human handoff.** A narrow set of deterministic triggers — late delivery,
non-delivery, damage on arrival, an explicit request for a person, a mention of
Procon — routes the message to a handoff instead of an answer. A late order or
damaged goods has no correct answer in the data, so any generated text would be
an empty promise. The web UI shows the trigger list and lets the customer switch
escalation off.

Every question is routed to catalog, order, policy, exchange, checkout or
handoff, and the CLI and web UI share that one path.

See [examples/conversations.md](examples/conversations.md) for the required
scenarios — those transcripts are generated by running the agent against live
Gemini, not written by hand.

## Use of code assistants

Claude Code (Claude Opus, in the VS Code extension) was used throughout, as a
pair programmer rather than a code generator: I described the intent and the
constraint, reviewed each diff, and kept the architectural decisions — the
structured/RAG split, the cheap model, the narrow handoff triggers — my own. It
was most useful for mechanical breadth: sweeping the retrieval helpers, keeping
the regex sets and their readable UI labels in sync, and running the unit tests
and the conversation evaluation after each change so regressions surfaced
immediately.

Generating the example transcripts by running the agent, instead of writing them
by hand, is what exposed three real bugs: a privacy refusal being rewritten away
by the model, a budget figure parsed as a model number, and a prompt-injection
attempt answered with an unrelated message. Each is now covered by a test.

## Known limitations

The database is a local snapshot and there is no real customer authentication —
`--customer-id` stands in for a verified session. The embedding index is rebuilt
at process start instead of persisting to pgvector/Supabase. The human handoff
detects and records the case, but dispatch to a support queue does not exist
yet, and checkout collects the purchase data without creating an order.

With more time I would persist chunks and embeddings, add incremental updates,
observability, and continuous recall/grounding evaluation.
