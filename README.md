# Empório da Música — grounded Gemini chatbot

A small Python customer-service agent for the Artefact AI Engineer challenge. It
answers in Brazilian Portuguese and covers catalog, prices, stock, promotions,
orders, delivery and store policies.

The design follows the useful split from TwinTweaker:

- CSV facts are retrieved deterministically. Product names, prices, stock,
  promotions and order data never come from vector similarity.
- The policy PDF is split into overlapping chunks and searched with hybrid
  keyword + Gemini embedding retrieval.
- gemini-3.1-flash writes only the short conversational layer over retrieved
  evidence. Without a key, deterministic offline answers remain available.

## Fastest evaluation on Linux

The repository includes one Linux runner. It creates .venv, installs
dependencies, runs the unit/frontend tests, and runs the multi-turn evaluation:

~~~bash
bash ./run.sh
~~~

Expected result:

~~~text
Ran 55 tests ... OK
The evaluator covers 62 source-backed turns; run it to refresh tests_ai/report.md.
~~~

The generated evaluation files are tests_ai/report.md and
tests_ai/results.json.

The runner also supports:

~~~bash
bash ./run.sh web    # start the browser UI
bash ./run.sh cli    # start the interactive CLI
bash ./run.sh live   # run the Gemini-backed evaluation
~~~

For the UI, open http://127.0.0.1:8000. The browser session keeps one
conversation history and sends its conversation id to the same StoreAgent
used by the CLI and tests.

The automation provided in this repository targets Linux. It requires Bash and
Python 3.10+; no Windows runner is included.

## Optional Gemini configuration

The application runs without a key. To enable Gemini, copy the example file and
fill in the token:

~~~bash
cp .env.example .env
~~~

~~~env
GOOGLE_GENERATIVE_AI_API_KEY=your_token
GEMINI_MODEL=gemini-3.1-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
~~~

.env is ignored by Git. The key is never stored in source code.

Direct commands, after the runner has created the environment:

~~~bash
./.venv/bin/python app.py --no-llm
./.venv/bin/python app.py --message "Quais violões Yamaha estão disponíveis?"
./.venv/bin/python app.py --customer-id 2 --message "Qual o status do pedido 8?"
./.venv/bin/python app.py --web --no-llm
~~~

## Architecture

Only app.py is a root application entrypoint. The implementation is split
into four focused modules:

~~~text
app.py          CLI entrypoint and compatibility exports
run.sh          Linux setup, test, evaluation and run commands
src/
  catalog.py    CSV loading, exact filters, joins and shared data models
  rag.py        Gemini REST client and policy-PDF hybrid RAG
  agent.py      routing, history, checkout/exchange state and answer guards
  web.py        dependency-free local browser UI
data/           catalog CSVs, policy PDF and other source data
tests/          deterministic unit and HTTP frontend tests
tests_ai/       multi-turn source-backed evaluation and report
~~~

There is no framework-specific agent abstraction. This keeps the retrieval and
grounding decisions visible to a reviewer.

## Retrieval and conversation flow

~~~text
message + previous turns
          |
          v
  deterministic router
     /             catalog route    policy route
     |               |
exact CSV query   PDF chunks:
(products,        keywords + embeddings
promotions,       (when Gemini is enabled)
orders)
                  /
      trusted grounding
             |
             v
   Gemini short rewrite
   or deterministic fallback
~~~

### Catalog path

Catalog questions use structured filters and joins:

1. normalize accents, common typos and singular/plural forms;
2. identify category, brand, model, budget and availability;
3. query the CSV-backed CatalogStore;
4. build exact grounding rows from the selected products;
5. format every returned price and stock value deterministically.

The LLM does not choose which catalog rows to show, change a price, or truncate a
requested complete list. For broad category searches the response can show a
small preview and a total; follow-ups such as 'mostre os outros', 'só esses?' or
'E da Tagima?' reuse the previous catalog context and resolve the next list
deterministically.

### Policy path

The policy manual is loaded from data/, extracted page by page and split into
1,400-character chunks with 180 characters of overlap. Retrieval combines:

- lexical matching, useful for exact rules, dates, percentages and terms;
- Gemini gemini-embedding-001 semantic similarity when a key is available;
- keyword fallback when there is no key or the network is unavailable.

Only retrieved policy context is sent to Gemini for prose generation. If no
relevant evidence exists, the agent returns a deterministic confirmation-needed
answer instead of inventing a rule.

### History and safety

Each turn is appended to the agent history. Recent user/model turns are replayed
for contextual follow-ups, while structured state tracks:

- the active catalog query and pagination;
- exchange product/date;
- checkout product, quantity, customer/contact data, payment installments and delivery address.

Customer and order tables are loaded only for an authenticated order lookup.
Checkout does not create an order. Human-handoff triggers such as a late,
undelivered or damaged order are deterministic and can be disabled in the UI.

## Tests

The test suite checks isolated behavior and realistic conversation flow:

~~~bash
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python tests_ai/run_evaluation.py
~~~

The evaluation compares answers with committed CSV/PDF source data. It checks
exact catalog rows, totals, promotions, policy evidence, history, privacy,
checkout and handoff behavior instead of only checking that a response sounds
plausible. The committed report is a previous run; the command above regenerates
it with the current 62-turn fixture set.

## Scope and limitations

This is a local prototype, **read-only by design**: it never writes to
`orders.csv`, `order_items.csv` or `customers.csv`. Every run starts from the
same committed snapshot, customer id stands in for authentication, embeddings
are rebuilt at process start, checkout collects and validates the full purchase
case but does not persist it, and a handoff records a ticket without dispatching
it. A production version would also persist the index, add real authentication
and observability.

**Turning this into a fully autonomous system — one that actually creates
orders and customer records as it talks to people — is the highest-priority
next step, not something this submission attempts.** The plan: four typed MCP
tools (`customers_find`, `customers_create`, `stock_check`, `orders_create`)
that own every write, called only after explicit customer confirmation. The
write is one transaction — find/create the customer, check stock, append to
`orders.csv` and `order_items.csv`, decrement stock — atomic (temp file plus
replace), single-writer (the web UI is threaded), and idempotent via a key
derived from the conversation, so a retry returns the existing order instead of
creating a second one. Totals, delivery estimate and tracking code still come
only from the data and the policy manual, never from the model. This was
deliberately left unbuilt: a rushed write path into shared CSVs the night
before a deadline is a worse risk than an honest "not yet."