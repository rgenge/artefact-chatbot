# Empório da Música — agente Gemini

Protótipo Python do agente de atendimento pedido no desafio. A arquitetura segue o processo do TwinTweaker:

1. O catálogo, promoções e pedidos são consultados de forma estruturada nos CSVs.
2. O manual em PDF é dividido em chunks de 1.400 caracteres com overlap de 180.
3. O PDF usa recuperação híbrida: keywords/ranking determinístico + embeddings Gemini (gemini-embedding-001, 768 dimensões).
4. O contexto recuperado é enviado ao gemini-3.1-flash para escrever respostas abertas curtas e acolhedoras.
5. Listas estruturadas de catálogo permanecem determinísticas: o modelo não pode resumir, truncar ou inventar linhas de preço/estoque.
6. Sem chave ou com falha de rede, o fallback local continua funcionando sem inventar preço ou estoque.

## Configuração

~~~bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
~~~

Preencha no .env:

~~~env
GOOGLE_GENERATIVE_AI_API_KEY=seu_token
GEMINI_MODEL=gemini-3.1-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
~~~

A chave não fica no código. O .env já está no .gitignore.

## Uso

Com Gemini ativado:

~~~bash
python app.py
~~~

Consulta direta:

~~~bash
python app.py --message "Quanto custa o Takamine GD20?"
~~~

Consulta de pedido com identificação:

~~~bash
python app.py --customer-id 2 --message "Qual o status do pedido 8?"
~~~

Modo offline, sem chamadas externas:

~~~bash
python app.py --no-llm
python -m unittest discover -s tests -v
~~~

Avaliação de conversas reais, offline e reproduzível:

~~~bash
python tests_ai/run_evaluation.py
~~~

O relatório é salvo em `tests_ai/report.md` e o resultado detalhado em `tests_ai/results.json`.

## Decisão técnica

O catálogo não é tratado como texto RAG: preço, estoque, status, promoção e itens de pedido precisam de filtros e joins exatos. Essa é a mesma separação do TwinTweaker para dados tabulares. O agente corrige typos comuns de marca, informa o total quando mostra uma prévia de cinco itens e mantém contexto em perguntas como “só esses?” e “E da Tagima?”.

O PDF segue RAG híbrido. Keywords preservam termos exatos, códigos e regras; embeddings recuperam perguntas semanticamente parecidas. Consultas de preço/modelo/política priorizam a evidência lexical, enquanto perguntas narrativas podem usar similaridade semântica. O Gemini só redige a resposta sobre o contexto recuperado; um guard determinístico rejeita respostas que introduzam valores monetários ausentes do contexto.

O histórico recente é reenviado ao Gemini como turnos user/model. O prompt limita escopo, tom, idioma, tamanho e proíbe inventar dados. Pedidos de acessórios são redirecionados conforme o manual.

Veja [examples/conversations.md](examples/conversations.md) para os cenários exigidos.

## Limitações

A base é um snapshot local e não há autenticação real de cliente. O índice de embeddings é reconstruído ao iniciar o processo, em vez de persistir em pgvector/Supabase como no TwinTweaker. Em produção eu persistiria chunks e embeddings, adicionaria atualização incremental, observabilidade, handoff humano e avaliação contínua de recall/grounding.


## Como o RAG e o catálogo trabalham juntos

O fluxo separa duas fontes de verdade. O catálogo (produtos, categorias, promoções, clientes e pedidos) é carregado dos CSVs e consultado com filtros, normalização e joins determinísticos. Assim, preço, estoque, desconto e status de pedido não são adivinhados por similaridade semântica.

O manual de políticas é extraído do PDF e dividido em chunks de 1.400 caracteres, com overlap de 180. Cada chunk pode ser recuperado por palavras-chave (termos exatos) ou por similaridade de embeddings Gemini (perguntas com redação diferente). O ranking híbrido envia os melhores chunks ao Gemini 3.1 Flash junto com o histórico. Uma validação local bloqueia valores monetários ausentes do contexto.

A UI (`python app.py --web`, em `http://127.0.0.1:8000`) usa exatamente o mesmo `StoreAgent`: cada pergunta é roteada para catálogo, pedido ou política sem duplicar a lógica.
