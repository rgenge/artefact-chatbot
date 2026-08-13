# Conversation examples

The examples use the CLI in offline mode (`--no-llm`). Prices and stock are read
from the CSVs; the rules come from the chunked PDF.

The agent answers in Brazilian Portuguese because the store is in Campo
Grande/MS and that is its persona. These are real transcripts, kept verbatim.

## 1. Catalog search with a structured filter

**Customer:** Quais opções de violões disponíveis custando até R$1000?

**Agent:** Encontrei violões disponíveis até R$ 1.000,00:

- Tagima Memphis AC-39 Nylon Natural: R$ 429,90; 15 em estoque.
- Rozini RC-104 Nylon Estudante: R$ 459,90; 10 em estoque.
- Shelby SN-100 Folk Aço Preto: R$ 549,00; 5 em estoque.
- Giannini GF-1R Aço Folk Sunburst: R$ 579,90; 7 em estoque.
- Yamaha C40 Nylon Natural: R$ 599,90; 12 em estoque.

## 2. Exact product lookup

**Customer:** Quanto custa o Takamine GD20?

**Agent:** Takamine GD20: R$ 2.199,00. Temos 5 unidade(s) em estoque.

## 3. Policy application

**Customer:** Me arrependi da minha compra, posso devolver meu pedido?

**Agent:** Em compras online, você pode pedir devolução em até 7 dias corridos após o recebimento, sem justificativa. O produto deve estar sem uso e na embalagem original; o reembolso ocorre na forma de pagamento original em até 10 dias úteis. (Manual de Políticas, p. 4, 5.)

## 4. Order lookup with customer identification

    $ python app.py --customer-id 2 --no-llm
    Você: Qual o status do pedido 8?
    Agente: O pedido 8 está enviado. Previsão de entrega: 2026-02-28. Rastreamento: BRJL5544332BR. Itens: 1x Kala KA-C Concert Mogno.

Without `--customer-id`, the agent asks for identification before revealing any
order data.

## 5. Escalation to a human

The non-trivial case: the message names a product, so a naive router would treat
it as a catalog search. A deterministic trigger sends it to a human instead.

    $ python app.py --no-llm
    Você: Meu violão está atrasado
    Agente: Sinto muito pelo transtorno. Já estou acionando um atendente da equipe para assumir o seu caso (protocolo H0001). Para agilizar, me informe o número do pedido e o nome ou e-mail usado na compra. O retorno acontece em até 24 horas úteis, no horário de atendimento.

The ticket is recorded with the customer, the latest order and the recent
history. Dispatch to the support queue is not implemented yet.
