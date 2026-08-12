# Exemplos de conversas

Os exemplos usam a CLI em modo offline (--no-llm). Preços e estoque são consultados nos CSVs; as regras vêm do PDF em chunks.

## 1. Busca de catálogo com filtro estruturado

**Cliente:** Quais opções de violões disponíveis custando até R$1000?

**Agente:** Encontrei violões disponíveis até R$ 1.000,00:

- Tagima Memphis AC-39 Nylon Natural: R$ 429,90; 15 em estoque.
- Rozini RC-104 Nylon Estudante: R$ 459,90; 10 em estoque.
- Shelby SN-100 Folk Aço Preto: R$ 549,00; 5 em estoque.
- Giannini GF-1R Aço Folk Sunburst: R$ 579,90; 7 em estoque.
- Yamaha C40 Nylon Natural: R$ 599,90; 12 em estoque.

## 2. Consulta exata de produto

**Cliente:** Quanto custa o Takamine GD20?

**Agente:** Takamine GD20: R$ 2.199,00. Temos 5 unidade(s) em estoque.

## 3. Aplicação de política

**Cliente:** Me arrependi da minha compra, posso devolver meu pedido?

**Agente:** Em compras online, você pode pedir devolução em até 7 dias corridos após o recebimento, sem justificativa. O produto deve estar sem uso e na embalagem original; o reembolso ocorre na forma de pagamento original em até 10 dias úteis. (Manual de Políticas, p. 4, 5.)

## 4. Pedido com identificação do cliente

    $ python app.py --customer-id 2 --no-llm
    Você: Qual o status do pedido 8?
    Agente: O pedido 8 está enviado. Previsão de entrega: 2026-02-28. Rastreamento: BRJL5544332BR. Itens: 1x Kala KA-C Concert Mogno.

Sem --customer-id, o agente pede a identificação antes de revelar dados do pedido.


