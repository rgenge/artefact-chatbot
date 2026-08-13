# AI conversation evaluation

This folder contains a reproducible, source-backed evaluation of the chatbot.
It exercises the same StoreAgent used by the CLI and web UI, using multi-turn
conversations instead of isolated string tests.

The checks compare answers with:

- product, stock, price, promotion, customer, order, and order-item CSV data;
- the extracted text of data/políticas_da_loja.pdf;
- safety expectations such as not inventing prices or exposing an unrelated order.

Run the deterministic evaluation from the project directory:

~~~
python tests_ai/run_evaluation.py
~~~

The command writes:

- tests_ai/report.md — human-readable answer-by-answer report;
- tests_ai/results.json — machine-readable results.

Offline mode uses deterministic retrieval and does not call Gemini. To evaluate
the complete Gemini writing path, put the key in .env and run:

~~~
python tests_ai/run_evaluation.py --live
~~~

--live keeps the same assertions, but the generated wording can vary. A
successful test requires every hard fact to be present and no unsupported
monetary value to be introduced. “Warnings” identify policy details that are
accurate in the PDF but are not yet included in the concise answer.
