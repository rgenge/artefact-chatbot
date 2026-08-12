# Empório da Música — agente Gemini

Protótipo Python do agente de atendimento pedido no desafio. A arquitetura segue o processo do TwinTweaker:

1. O catálogo, promoções e pedidos são consultados de forma estruturada nos CSVs.
2. O manual em PDF é dividido em chunks de 1.400 caracteres com overlap de 180.
3. O PDF usa recuperação híbrida: keywords/ranking determinístico + embeddings Gemini (gemini-embedding-001, 768 dimensões).
4. O contexto recuperado é enviado ao gemini-3.1-flash para escrever uma resposta curta e acolhedora.
5. Sem chave ou com falha de rede, o fallback local continua funcionando sem inventar preço ou estoque.

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

## Decisão técnica

O catálogo não é tratado como texto RAG: preço, estoque, status, promoção e itens de pedido precisam de filtros e joins exatos. Essa é a mesma separação do TwinTweaker para dados tabulares.

O PDF segue RAG híbrido. Keywords preservam termos exatos, códigos e regras; embeddings recuperam perguntas semanticamente parecidas. Consultas de preço/modelo/política priorizam a evidência lexical, enquanto perguntas narrativas podem usar similaridade semântica. O Gemini só redige a resposta sobre o contexto recuperado; um guard determinístico rejeita respostas que introduzam valores monetários ausentes do contexto.

O histórico recente é reenviado ao Gemini como turnos user/model. O prompt limita escopo, tom, idioma, tamanho e proíbe inventar dados. Pedidos de acessórios são redirecionados conforme o manual.

Veja [examples/conversations.md](examples/conversations.md) para os cenários exigidos.

## Limitações

A base é um snapshot local e não há autenticação real de cliente. O índice de embeddings é reconstruído ao iniciar o processo, em vez de persistir em pgvector/Supabase como no TwinTweaker. Em produção eu persistiria chunks e embeddings, adicionaria atualização incremental, observabilidade, handoff humano e avaliação contínua de recall/grounding.

=======
# artefact-chatbot
Chatbot with RAG system using gemini 3.1 for a store.
>>>>>>> 94b085f10f7a775b715841b5a3ae8d1d6070f8a6
