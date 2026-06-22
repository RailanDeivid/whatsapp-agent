# Documentação Completa — WhatsApp Agent

> Última atualização: 23/04/2026

---

## Parte 1 — Contexto do Projeto

### O que é

O **WhatsApp Agent** é um assistente inteligente de dados integrado ao WhatsApp. Ele recebe mensagens de texto, áudio e imagem, interpreta a intenção do usuário via LLM e responde com dados operacionais em tempo real, documentos internos ou respostas gerais.

A arquitetura é **multi-agente**: um **Router LLM** classifica cada mensagem e roteia para o **Agente SQL** (Grok + Dremio/MySQL), o **Agente RAG** (Grok + ChromaDB) ou responde diretamente via LLM. Respostas gerais e mensagens de espera usam GPT-4o mini. Áudios são transcritos via Whisper. Imagens são interpretadas por um modelo com visão.

### Para que serve

- Consultar faturamento, vendas, ticket médio, fluxo de pessoas por casa/vertical/marca
- Consultar delivery por plataforma (iFood, Rappi, app próprio)
- Consultar formas de pagamento, estornos, cortesias e metas (orçado vs realizado)
- Consultar pedidos de compra, fornecedores e notas fiscais (MySQL)
- Buscar políticas, procedimentos, organogramas e contatos internos (RAG)
- Gerar gráficos (PNG) e exportar dados em Excel (XLSX)
- Fazer resumo geral consolidado de uma casa/vertical/marca (5 queries em paralelo)

### Quem usa

Equipes internas de gestão, operações e finanças, via WhatsApp. Acesso controlado por lista de usuários autorizados com dois perfis: **usuário** e **admin**.

---

## Parte 2 — Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Backend | Python 3.13 + FastAPI 0.115 |
| LLM Principal (agentes) | Grok 4.1 Fast via OpenRouter |
| LLM Fallback / Geral | GPT-4o mini via OpenRouter |
| Visão (imagens) | GPT-4o mini via OpenRouter |
| Transcrição de áudio | Whisper-1 (OpenAI direto) |
| Embeddings RAG | text-embedding-ada-002 (OpenAI direto) |
| Framework de agentes | LangChain ReAct |
| Banco operacional | Dremio (REST API) |
| Banco de compras | MySQL |
| Vector DB | ChromaDB |
| Cache / Histórico | Redis 7 |
| Gateway WhatsApp | Evolution API |
| Infraestrutura | Docker Compose (4 serviços) |
| Controle de acesso | SQLite |

---

## Parte 3 — Arquitetura do Sistema

### Fluxo de uma mensagem

```
WhatsApp (usuário)
      ↓
Evolution API → POST /webhook
      ↓
app.py — valida acesso, rate limit, tipo de mídia (texto/áudio/imagem)
      ↓
message_buffer.py — debounce 3s (agrupa msgs rápidas) + "digitando..."
      ↓
chains.py — fast-paths ──→ roteador LLM → agente
      ↓
      ├── "sql"   → Agente SQL (Grok + tools Dremio/MySQL)
      ├── "docs"  → Agente RAG (Grok + ChromaDB)
      ├── "ambos" → Ambos em paralelo (ThreadPoolExecutor)
      └── "geral" → GPT-4o mini direto (sem tools)
      ↓
message_buffer.py — envia resposta (texto / PNG / XLSX) via Evolution API
      ↓
WhatsApp (usuário)
```

### Classificação de intenção (Router)

| Rota | Quando aciona | Modelo | Ferramentas |
|------|--------------|--------|------------|
| `sql` | Vendas, faturamento, delivery, pagamentos, estornos, metas, compras | Grok 4.1 Fast | Todas as tools Dremio + MySQL + gráfico + Excel |
| `docs` | Políticas, organograma, contatos, ramais, emails, procedimentos | Grok 4.1 Fast | `consultar_documentos` (Chroma) |
| `ambos` | Pergunta envolve dados numéricos E documentos | Grok 4.1 Fast | SQL + RAG em paralelo |
| `geral` | Saudações, agradecimentos, perguntas conceituais, fora do escopo | GPT-4o mini | Nenhuma (LLM direto) |

### Fast-paths (sem LLM)

Algumas situações são resolvidas antes de qualquer chamada ao LLM:

| Trigger | Resposta |
|---------|----------|
| Pergunta sobre bases disponíveis | Lista fixa das bases disponíveis |
| Palavra "salário/remuneração/folha" | Bloqueio — dado sensível |
| Pedido de tabela | Aviso que não suporta tabelas, oferece Excel |
| Pedido de Excel após consulta recente | Exporta direto do cache (sem nova query) |
| Saudação simples (1ª mensagem) | Resposta fixa de boas-vindas com lista de bases |
| Saudação simples (retornando) | GPT-4o mini |

---

## Parte 4 — Estrutura de Pastas

```
whatsapp-agent/
├── src/
│   ├── app.py                    # Servidor FastAPI, webhook, comandos admin
│   ├── chains.py                 # Roteador e executores dos agentes
│   ├── prompts.py                # System prompts de todos os agentes
│   ├── config.py                 # Variáveis de ambiente
│   ├── memory.py                 # Histórico de conversas (Redis)
│   ├── message_buffer.py         # Debounce de mensagens e indicadores de digitação
│   ├── access_control.py         # Controle de acesso por SQLite
│   ├── vectorstore.py            # Indexação ChromaDB para RAG
│   ├── docs/
│   │   └── architecture.svg      # Diagrama do fluxo completo
│   ├── connectors/
│   │   ├── dremio.py             # Cliente REST Dremio (com cache Redis)
│   │   └── mysql.py              # Pool de conexões MySQL (com cache Redis)
│   ├── integrations/
│   │   ├── evolution_api.py      # Envio/recebimento WhatsApp
│   │   └── transcribe.py         # Transcrição de áudio via Whisper
│   └── tools/
│       ├── dremio_tools.py       # Tools: transacoes, delivery, pagamentos, estornos, metas, cortesias
│       ├── resumo_tool.py        # Tool: resumo consolidado em paralelo
│       ├── mysql_tools.py        # Tool: consulta de compras
│       ├── rag_tool.py           # Tool: busca semântica em documentos
│       ├── chart_tool.py         # Tool: gráficos matplotlib/seaborn → PNG
│       ├── excel_tool.py         # Tool: exportação pandas → XLSX
│       ├── utils.py              # Helpers compartilhados
│       └── fantasia_abreviacao.py # Mapeamento abreviação → nome fantasia das casas
├── tests/
│   ├── conftest.py               # Setup de mocks e variáveis falsas para testes locais
│   ├── test_access_control.py
│   ├── test_app.py
│   ├── test_chains.py
│   ├── test_config.py
│   ├── test_evolution_api.py
│   ├── test_message_buffer.py
│   ├── test_resumo_tool.py
│   └── test_utils.py
├── rag_files/                    # PDFs/TXTs a indexar
│   └── processed/                # Arquivos já indexados (movidos para cá)
├── vectorstore/                  # Índice ChromaDB persistido em disco
├── data/
│   └── access.db                 # SQLite de controle de acesso
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env                          # Variáveis de ambiente (NÃO versionar)
├── .env.example                  # Template de variáveis
├── CLAUDE.md                     # Instruções para o assistente de código
└── documentação projeto.md       # Este arquivo
```

---

## Parte 5 — Arquivos e Códigos (detalhe por arquivo)

---

### `src/app.py` — Servidor principal

**Para que serve:** Ponto de entrada HTTP. Recebe webhooks da Evolution API, valida autenticação e rate limit, e roteia mensagens para o buffer. Também expõe endpoints administrativos.

**Endpoints:**

| Método | Rota | Auth | Descrição |
|--------|------|------|-----------|
| `GET` | `/health` | — | Checa Redis, MySQL e vectorstore |
| `GET` | `/metrics` | — | Cache hit rate, latência por bucket, erros |
| `POST` | `/webhook` | Evolution API | Recebe mensagens WhatsApp |
| `POST` | `/limpar_cache` | `x-api-key` | Remove respostas cacheadas |
| `POST` | `/reindexar` | `x-api-key` | Indexa novos PDFs/TXTs |

**Lógica do webhook (passo a passo):**
1. Ignora mensagens de grupos (`@g.us`) e do próprio bot (`fromMe: true`)
2. Valida se o usuário está autorizado em SQLite
3. Aplica rate limiting com pipeline Redis atômico
4. Intercepta confirmação pendente do comando `/limpar`
5. Roteia comandos admin (`/autorizar`, `/bloquear`, etc.) para `_handle_admin_command()`
6. Transcreve áudio com Whisper se não houver texto
7. Interpreta imagem com modelo de visão se não houver texto
8. Envia para `buffer_message()` para processamento assíncrono

**Comandos admin (via WhatsApp — só para admins):**

```
/autorizar PHONE ; Nome ; Cargo ; Casa [; admin]
/bloquear PHONE
/desbloquear PHONE
/remover PHONE
/atualizar PHONE_ANTIGO ; PHONE_NOVO
/usuarios [admin]
/historico PHONE [dias]
/limpar_usuario PHONE
/limpar              ← pede confirmação SIM/NÃO antes de executar
/reindexar
/ajuda
```

**O que pode ser ajustado:**
- `RATE_LIMIT_MAX` e `RATE_LIMIT_WINDOW` no `.env` — msgs por janela de tempo
- `SUPPORT_CONTACT` no `.env` — contato exibido para usuários não autorizados
- `UNAUTHORIZED_MESSAGE` no `.env` — mensagem completa de acesso negado

---

### `src/chains.py` — Roteador e agentes

**Para que serve:** Orquestra o fluxo de IA. Classifica intenção, executa o agente correto, aplica fast-paths e pós-processa a resposta.

**Funções principais:**
- `route_and_invoke(message, session_id, sender_name, on_thinking)` — ponto de entrada principal
- `_classify_intent(message, history_text)` — classifica intenção via LLM
- `_run_sql_agent(...)` — executa agente SQL com Grok
- `_run_rag_agent(...)` — executa agente RAG com Grok
- `_run_general_response(...)` — resposta direta com GPT-4o mini
- `generate_thinking_message(message)` — gera mensagem de espera criativa via LLM
- `extract_text_from_image(image_b64)` — interpreta imagem com modelo de visão

**Pós-processamento:**
```python
response = _strip_emojis(response)                 # Remove emojis
response = normalize_whatsapp_markdown(response)   # **bold** → *bold*
```

**Cache de respostas Redis:**
- Chave: `cache:{session_id}:{md5(mensagem normalizada)}`
- TTL: `QUERY_CACHE_TTL` segundos (padrão 300s = 5 min)
- Não cacheia respostas de erro

**Singleton thread-safe dos modelos:**
- Modelos e AgentExecutors são inicializados na primeira mensagem (lazy)
- Double-check locking com `threading.Lock()` por modelo

**O que pode ser ajustado:**
- `SQL_AGENT_MAX_ITERATIONS` no `.env` — máx iterações ReAct (padrão 8)
- `SQL_AGENT_MAX_EXECUTION_TIME` no `.env` — timeout do agente em segundos (padrão 600)
- `QUERY_CACHE_TTL` no `.env` — TTL do cache de respostas
- `CONVERSATION_MAX_HISTORY` no `.env` — msgs no contexto enviadas ao LLM
- `_THINKING_STARTERS` e `_THINKING_ENDINGS` no código — variações na mensagem de espera
- `_RESUMO_RE` no código — regex que detecta pedidos de resumo

---

### `src/prompts.py` — System prompts

**Para que serve:** Define o comportamento de todos os agentes via `PromptTemplate` do LangChain.

**Prompts disponíveis:**
- `react_prompt` — Agente SQL (ReAct, ~250 linhas de regras)
- `rag_prompt` — Agente RAG
- `router_prompt` — Roteador de intenções
- `general_prompt` — Respostas gerais (GPT-4o mini)

**Regras principais do Agente SQL (resumidas):**

| # | Regra |
|---|-------|
| 1 | Sempre responder em português |
| 2 | Nunca inventar dados — só usar o retorno das ferramentas |
| 3 | Sempre chamar as ferramentas para dados, mesmo se pergunta parecer igual a anterior |
| 4 | Mapeamento intenção → ferramenta (transacoes→consultar_transacoes, etc.) |
| 5-8 | Formatação: BRL, datas, listas, percentuais |
| 9 | Em erro de ferramenta: responder "Tive um problema técnico" |
| 10-15 | Formatos específicos: metas, SSS, cortesias, dias da semana |
| 16-20 | Sintaxe Dremio: datas, GROUP BY, granularidade temporal |
| 21 | Resumo: SOMENTE com "resumo", "visão geral" ou "panorama" |

**Atenção técnica:** JSON nos exemplos usa `{{}}` (duplo) para escapar do PromptTemplate.

**O que pode ser ajustado:**
- Qualquer regra numerada para mudar comportamento do agente
- Tom e formato nas regras 5-8
- Exemplos de SQL nas descriptions das tools em `dremio_tools.py`

---

### `src/config.py` — Configuração

**Para que serve:** Carrega e valida todas as variáveis de ambiente. A função `_require()` falha na inicialização se variável obrigatória estiver ausente.

**Variáveis obrigatórias** (falham no boot se ausentes):
```
EVOLUTION_API_URL, EVOLUTION_INSTANCE_NAME, AUTHENTICATION_API_KEY
ROUTER_API_KEY, ROUTER_MODEL_NAME, ROUTER_MODEL_TEMPERATURE
BOT_REDIS_URI
DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
DREMIO_HOST, DREMIO_USER, DREMIO_PASSWORD
```

**Variáveis opcionais com defaults:**
```
FALLBACK_MODEL_NAME        — modelo fallback (GPT-4o mini)
VISION_MODEL_NAME          — modelo de visão (default = ROUTER_MODEL_NAME)
WHISPER_API_KEY            — chave OpenAI para Whisper/Embeddings
QUERY_CACHE_TTL=300        — TTL cache Redis (segundos)
RATE_LIMIT_MAX=10          — máx mensagens por janela
RATE_LIMIT_WINDOW=60       — janela de rate limit (segundos)
SQL_AGENT_MAX_ITERATIONS=8
SQL_AGENT_MAX_EXECUTION_TIME=600
DREMIO_MAX_CONCURRENT=3    — queries paralelas simultâneas
DREMIO_MAX_ROWS=50000      — limite de linhas por query
DREMIO_POLL_INITIAL=2      — intervalo inicial de polling (s)
DREMIO_POLL_MAX=30         — intervalo máximo de polling (s)
MYSQL_POOL_SIZE=5          — conexões simultâneas no pool MySQL
EXCEL_TTL=300              — TTL do arquivo Excel no Redis
DEBOUNCE_SECONDS=3         — tempo de debounce de mensagens
BUFFER_TTL=300             — TTL do buffer de mensagens no Redis
CONVERSATION_MAX_HISTORY=5 — msgs de histórico enviadas ao LLM
SEED_USERS                 — usuários iniciais (PHONE:NOME:CARGO:CASA[:admin])
SUPPORT_CONTACT            — contato de suporte para mensagem de acesso negado
```

---

### `src/memory.py` — Histórico de conversas

**Para que serve:** Gerencia o histórico de mensagens de cada usuário no Redis usando `RedisChatMessageHistory` do LangChain.

**Armazenamento:**
- Chave Redis: `message_store:{session_id}`
- TTL: 10 dias (`_SESSION_TTL = 864000`)
- `session_id` = `5511999999999@s.whatsapp.net`

**Funções:**
- `get_session_history(session_id)` — objeto de histórico LangChain
- `get_session_messages(session_id, since_ts)` — lista de msgs (opcionalmente filtradas por timestamp)
- `clear_session(session_id)` — apaga uma sessão
- `clear_all_sessions()` — apaga tudo via **Lua script atômico** (evita race condition)

**O que pode ser ajustado:**
- `_SESSION_TTL` no código — duração do histórico
- `CONVERSATION_MAX_HISTORY` no `.env` — quantas msgs são enviadas como contexto

---

### `src/message_buffer.py` — Buffer e debounce

**Para que serve:** Agrupa mensagens rápidas em sequência, gerencia o indicador "digitando..." e envia a resposta correta (texto, PNG ou XLSX).

**Fluxo:**
1. Mensagem chega → adiciona ao buffer Redis
2. Cria (ou reseta) task assíncrona com delay de `DEBOUNCE_SECONDS`
3. Após o delay, lê todo o buffer e concatena as mensagens
4. Executa `route_and_invoke()` em thread separada (não bloqueia o event loop)
5. Durante o processamento, envia "digitando..." a cada 3s via Evolution API
6. Ao receber resposta, detecta se contém marker de gráfico `[CHART:...]` ou Excel `[EXCEL:...]`
7. Envia texto + arquivo de mídia

**Concorrência segura:**
- `_debounce_lock = asyncio.Lock()` protege o dict `debounce_tasks` contra race conditions
- Cleanup da task no `finally` só executa se for a task atual da sessão

**Cancelamento (usuário pode digitar):**
- Palavras exatas: `cancela`, `cancel`, `cancelar`, `pare`, `parar`, `stop`, `esquece`, `chega`, ...
- Frases: `não quero mais`, `esquece isso`, `deixa pra lá`, `cancela tudo`, ...

**O que pode ser ajustado:**
- `DEBOUNCE_SECONDS` no `.env` — tempo de espera antes de processar
- `_CANCEL_EXACT` e `_CANCEL_PHRASES` no código — palavras de cancelamento
- `_CANCEL_RESPONSES` no código — respostas ao cancelamento

---

### `src/access_control.py` — Controle de acesso

**Para que serve:** Gerencia quais usuários podem usar o bot via SQLite.

**Schema da tabela `authorized_users`:**
```sql
telefone TEXT PRIMARY KEY
nome TEXT, cargo TEXT, casa TEXT
is_admin INTEGER (0=usuário, 1=admin)
active INTEGER (0=bloqueado, 1=ativo)
added_by_tel TEXT, added_by_nome TEXT
created_at TEXT, updated_at TEXT
```

**Funções:**
- `is_authorized(phone)` — ativo no banco?
- `is_admin(phone)` — é admin e está ativo?
- `authorize(...)` — cadastra ou reativa usuário
- `revoke(...)` — bloqueia (soft delete — mantém registro)
- `unblock(...)` — desbloqueia
- `delete_user(...)` — remove permanentemente
- `update_phone(old, new)` — migra número
- `list_users()` — lista todos

**Seed de usuários (`SEED_USERS` no `.env`):**
```
5511999000000:João Silva:Analista:Matriz:admin,5511888000000:Maria:Gerente:SP
```
- Inseridos no boot (`init_db()`) apenas se não existirem
- Nunca sobrescreve usuários já cadastrados

**O que pode ser ajustado:**
- `SEED_USERS` no `.env` — administradores iniciais
- `SQLITE_PATH` no `.env` — localização do banco

---

### `src/vectorstore.py` — Indexação RAG

**Para que serve:** Gerencia o índice vetorial ChromaDB para busca semântica em documentos.

**Fluxo de indexação:**
1. Admin coloca PDF ou TXT em `rag_files/`
2. Chama `/reindexar` ou comando `/reindexar` no WhatsApp
3. `load_documents()` carrega o arquivo com `PyPDFLoader` ou `TextLoader`
4. Divide em chunks de 1000 chars (overlap 200) com `RecursiveCharacterTextSplitter`
5. Gera embeddings com `text-embedding-ada-002`
6. Adiciona ao índice ChromaDB persistido em `vectorstore/`
7. Move o arquivo para `rag_files/processed/` (não deleta)

**Funções:**
- `get_vectorstore()` — carrega índice existente ou cria novo
- `reload_vectorstore()` — indexa novos arquivos sem reiniciar (protegido por `threading.Lock`)

**O que pode ser ajustado:**
- `chunk_size=1000` e `chunk_overlap=200` no código
- `RAG_FILES_DIR` no `.env` — pasta dos arquivos
- `VECTOR_STORE_PATH` no `.env` — onde o índice é salvo

---

### `src/connectors/dremio.py` — Cliente Dremio

**Para que serve:** Executa queries SQL no Dremio via REST API com autenticação, polling, paginação e cache Redis.

**Fluxo de uma query:**
1. Verifica cache Redis (`qcache:{md5_da_query}`) → retorna imediatamente se hit
2. Obtém token de autenticação (cacheado por 55 min)
3. Submete SQL via `POST /api/v3/sql`
4. Polling do status com backoff exponencial (`DREMIO_POLL_INITIAL` → `DREMIO_POLL_MAX`)
5. Lê resultados paginados (500 linhas por página)
6. Cacheia DataFrame no Redis com TTL `QUERY_CACHE_TTL`
7. Retorna DataFrame

**Assinatura:** `client(sql: str, max_wait: int = 360) -> pd.DataFrame`

**O que pode ser ajustado:**
- `max_wait=360` no caller — timeout por query (resumo usa 600s)
- `DREMIO_POLL_INITIAL=2` e `DREMIO_POLL_MAX=30` no `.env` — polling
- `DREMIO_MAX_ROWS=50000` no `.env` — limite de linhas
- `QUERY_CACHE_TTL=300` no `.env` — TTL do cache

---

### `src/connectors/mysql.py` — Cliente MySQL

**Para que serve:** Executa queries no MySQL com pool de conexões, retry e cache Redis.

**Erros permanentes (sem retry):**

| Código | Significado |
|--------|-------------|
| 1064 | Erro de sintaxe SQL |
| 1045 | Acesso negado |
| 1146 | Tabela não existe |
| 1049 | Banco não existe |
| 1054 | Coluna não existe |
| 1142 | Permissão negada |

**O que pode ser ajustado:**
- `MYSQL_POOL_SIZE=5` no `.env` — conexões simultâneas
- `RETRY_MAX_ATTEMPTS=3` no `.env` — tentativas antes de desistir
- `RETRY_BACKOFF_BASE=2` no `.env` — base exponencial (2s, 4s, 8s...)

---

### `src/integrations/evolution_api.py` — Gateway WhatsApp

**Para que serve:** Interface com a Evolution API para enviar/receber todo tipo de mídia.

**Funções:**
- `send_whatsapp_message(number, text)` — envia texto (divide se > 3000 chars)
- `send_whatsapp_image(number, b64, caption)` — envia PNG em base64
- `send_whatsapp_document(number, b64, filename)` — envia XLSX
- `send_whatsapp_presence(number)` — envia "digitando..."
- `send_whatsapp_reaction(number, message_id, emoji)` — reage a mensagem
- `get_media_base64(message_key)` — baixa mídia (áudio/imagem) em base64

**Divisão de mensagens longas:**
- Limite: `_MAX_MSG_LEN = 3000` chars por mensagem
- Cap: `_MAX_CHUNKS = 10` partes máximas (acima disso trunca com aviso de Excel)
- Ordem de quebra: parágrafo (`\n\n`) → linha (`\n`) → corte duro por caracteres
- Delay entre partes: `_CHUNK_DELAY = 2.0` segundos
- Retry por parte: `_SEND_RETRIES = 3` tentativas

**O que pode ser ajustado:**
- `_MAX_MSG_LEN` — limite por mensagem
- `_MAX_CHUNKS` — máximo de partes
- `_CHUNK_DELAY` — pausa entre partes (evita rate limit da Evolution)

---

### `src/integrations/transcribe.py` — Transcrição de áudio

**Para que serve:** Converte áudio OGG do WhatsApp em texto com Whisper-1.

**Fluxo:**
1. Recebe áudio em base64
2. Decodifica para bytes
3. Envia para `openai.audio.transcriptions.create(model="whisper-1", language="pt")`
4. Retorna texto transcrito

**O que pode ser ajustado:**
- `language="pt"` — remover para detecção automática de idioma
- `WHISPER_API_KEY` no `.env`

---

### `src/tools/dremio_tools.py` — Ferramentas SQL Dremio

**Para que serve:** Define as 6 ferramentas LangChain do Agente SQL para consultar dados no Dremio.

**Semáforo de concorrência:**
- `threading.Semaphore(DREMIO_MAX_CONCURRENT)` — limita queries paralelas
- Timeout de 60s na fila — se não conseguir vaga, retorna erro ao usuário
- Notifica o usuário via WhatsApp ao entrar na fila de espera

**Ferramentas e tabelas:**

| Ferramenta | Tabela Dremio | Coluna de data | Uso |
|-----------|--------------|---------------|-----|
| `consultar_transacoes` | `fTransacoes` | `data_evento` | Faturamento, TM, fluxo, mix, SSS |
| `consultar_delivery` | `fDelivery` | `data_evento` | Delivery por plataforma |
| `consultar_formas_pagamento` | `fFormasPagamento` | `data` | Mix de pagamentos |
| `consultar_estornos` | `fEstornos` | `data_evento` | Cancelamentos e devoluções |
| `consultar_metas` | `dMetas` | `DATA` | Realizado vs orçado |
| `consultar_cortesias` | `fCortesias` | `data_evento` | Itens cortesia |

**Retry automático:**
- 2 tentativas por query
- Em timeout na 1ª tentativa, notifica o usuário e tenta novamente

**O que pode ser ajustado:**
- `DREMIO_MAX_CONCURRENT=3` no `.env`
- `_SEMAPHORE_TIMEOUT=60` no código
- Descriptions das tools — exemplos de SQL, colunas disponíveis

---

### `src/tools/resumo_tool.py` — Resumo consolidado

**Para que serve:** Gera resumo completo de uma casa/vertical/marca disparando 5 queries no Dremio **em paralelo**.

**Ativação:** SOMENTE com as palavras exatas `"resumo"`, `"visão geral"` ou `"panorama"` na mensagem.

**Queries em paralelo (5 threads simultâneas):**

| Seção | Tabela | O que traz |
|-------|--------|-----------|
| Vendas + Mix | `fTransacoes` | Faturamento, TM, fluxo, mix por Grande_Grupo |
| Delivery | `fDelivery` | Faturamento por plataforma |
| Formas de pagamento | `fFormasPagamento` | Receita por forma de pagamento |
| Estornos | `fEstornos` | Total estornado + quantidade de itens |
| Cortesias | `fCortesias` | Total em cortesias |

**Timeout:** 580s para o conjunto todo (cada query individual: `max_wait=600`)

**Validações de entrada:**
- Formato `AAAA-MM-DD` das datas
- Data início não pode ser posterior à data fim

**Output:** Mensagem formatada com `*negrito*` para WhatsApp, blocos separados por `\n\n`.

---

### `src/tools/mysql_tools.py` — Ferramenta de compras

**Para que serve:** Consulta pedidos de compra, fornecedores e notas fiscais no MySQL.

**Filtros de segurança:**
- Lista de CNPJs bloqueados (fornecedores internos/sistema) para não contaminar resultados
- Abreviações de casas expandidas automaticamente na query antes de executar

**O que pode ser ajustado:**
- CNPJs bloqueados no código — adicionar/remover conforme necessário
- Mapeamento de abreviações em `fantasia_abreviacao.py`

---

### `src/tools/rag_tool.py` — Busca em documentos

**Para que serve:** Busca semanticamente no ChromaDB. Retorna os `k` trechos mais relevantes para a pergunta.

**Parâmetros de busca:**
- `k=5` — 5 trechos mais relevantes

**O que pode ser ajustado:**
- `k=5` no código — aumentar para respostas mais completas (mais lento)

---

### `src/tools/chart_tool.py` — Geração de gráficos

**Para que serve:** Gera gráficos PNG a partir de uma query SQL usando matplotlib/seaborn.

**Tipos de gráfico:**
- `barra` — default, horizontal se > 6 itens
- `linha` — evolução temporal
- `pizza` — participação percentual (fatias < 2% agrupadas em "Outros")

**Fluxo:**
1. Recebe JSON: `sql`, `titulo`, `col_categoria`, `col_valor`, `tipo`, `fonte`
2. Executa query no Dremio (`fonte=dremio`) ou MySQL (`fonte=mysql`)
3. Renderiza gráfico com paleta verde
4. Converte para PNG base64
5. Salva no Redis com TTL de 120s
6. Retorna marker `[CHART:key|caption:titulo]`
7. `message_buffer.py` detecta, lê base64 e envia a imagem

**Cleanup:** `plt.close("all")` em bloco `finally` evita memory leak.

**O que pode ser ajustado:**
- `_CHART_TTL=120` no código — TTL do PNG no Redis
- `_THEME["dpi"]=160` — qualidade da imagem
- Paleta `_GREEN_PALETTE` e `_pie_palette()` — cores

---

### `src/tools/excel_tool.py` — Exportação Excel

**Para que serve:** Exporta dados para XLSX e envia via WhatsApp.

**Fast-path:** se já existe DataFrame em cache da sessão com ≥4 colunas, exporta direto sem nova query.

**Fluxo:**
1. Verifica cache `lastdf:{session_id}` no Redis
2. Se hit com ≥4 colunas: exporta direto
3. Se não: executa nova query SQL
4. Gera XLSX com pandas `ExcelWriter`
5. Salva base64 no Redis com TTL `EXCEL_TTL`
6. Retorna marker `[EXCEL:key|caption:filename]`

---

### `src/tools/utils.py` — Utilitários compartilhados

**Para que serve:** Funções auxiliares usadas por múltiplos módulos.

| Função | O que faz |
|--------|----------|
| `strip_markdown(query)` | Remove ` ``` ` e JSON wrappers antes do SQL |
| `extract_json(text)` | Parseia JSON tolerante (trailing commas, aspas simples) |
| `fmt_brl(value)` | `1234.56` → `"R$ 1.234,56"` |
| `fmt_int_br(value)` | `1000` → `"1.000"` |
| `normalize_whatsapp_markdown(text)` | `**bold**` → `*bold*`, remove headings `#` |
| `format_df(df)` | DataFrame → texto legível para o LLM |

---

### `src/tools/fantasia_abreviacao.py` — Mapeamento de casas

**Para que serve:** Dicionário `ABREVIACAO_TO_FANTASIA` com o de/para entre código abreviado e nome fantasia exato do banco de dados.

**Exemplos:**
```python
"TBI"    → "UNIDADE L ITAIM"
"LOJA BH" → "UNIDADE G BELO HORIZONTE"
"BPI"    → "UNIDADE A ITAIM"
```

**Onde é usado:**
- `dremio_tools.py` — gera hint no prompt com todos os códigos válidos
- `mysql_tools.py` — expande abreviações na query antes de executar

**Para adicionar uma nova casa:** editar apenas este arquivo. Sem alterar código.

---

## Parte 6 — Configuração Completa (.env)

```env
# ── Evolution API ──────────────────────────────────────────────────
EVOLUTION_API_URL=http://evolution-api:8080
EVOLUTION_INSTANCE_NAME=nome-da-instancia
AUTHENTICATION_API_KEY=sua-api-key

# ── OpenRouter (LLMs) ──────────────────────────────────────────────
ROUTER_API_KEY=sk-or-v1-...
ROUTER_BASE_URL=https://openrouter.ai/api/v1
ROUTER_MODEL_NAME=x-ai/grok-4.1-fast
ROUTER_MODEL_TEMPERATURE=0
FALLBACK_MODEL_NAME=openai/gpt-4o-mini
VISION_MODEL_NAME=openai/gpt-4o-mini

# ── OpenAI (Whisper + Embeddings) ──────────────────────────────────
WHISPER_API_KEY=sk-...

# ── Redis ──────────────────────────────────────────────────────────
BOT_REDIS_URI=redis://redis:6379/0
DEBOUNCE_SECONDS=3
BUFFER_TTL=300
QUERY_CACHE_TTL=300

# ── MySQL (externo ao Docker) ──────────────────────────────────────
DB_HOST=host-mysql
DB_PORT=3306
DB_USER=usuario
DB_PASSWORD=senha
DB_NAME=banco

# ── Dremio (externo ao Docker) ─────────────────────────────────────
DREMIO_HOST=host:9047
DREMIO_USER=usuario
DREMIO_PASSWORD=senha
DREMIO_POLL_INITIAL=2
DREMIO_POLL_MAX=30
DREMIO_MAX_CONCURRENT=3
DREMIO_MAX_ROWS=50000

# ── RAG ────────────────────────────────────────────────────────────
RAG_FILES_DIR=rag_files
VECTOR_STORE_PATH=vectorstore

# ── Agentes ────────────────────────────────────────────────────────
SQL_AGENT_MAX_ITERATIONS=8
SQL_AGENT_MAX_EXECUTION_TIME=600
RAG_AGENT_MAX_ITERATIONS=4
RAG_AGENT_MAX_EXECUTION_TIME=60
CONVERSATION_MAX_HISTORY=5

# ── Rate limiting ──────────────────────────────────────────────────
RATE_LIMIT_MAX=10
RATE_LIMIT_WINDOW=60

# ── Controle de acesso ─────────────────────────────────────────────
SQLITE_PATH=data/access.db
SEED_USERS=5511999990000:João Silva:TI:Matriz:admin
SUPPORT_CONTACT=administrador
UNAUTHORIZED_MESSAGE=Olá! Você não está autorizado. Entre em contato com um administrador.

# ── Misc ───────────────────────────────────────────────────────────
MYSQL_POOL_SIZE=5
EXCEL_TTL=300
RETRY_MAX_ATTEMPTS=3
RETRY_BACKOFF_BASE=2
```

---

## Parte 7 — Modelos de IA

### Modelos em uso

| Componente | Modelo | Provider |
|------------|--------|----------|
| Agente SQL + RAG + Router | Grok 4.1 Fast | OpenRouter |
| Fallback SQL + RAG | GPT-4o mini | OpenRouter |
| Respostas gerais + thinking message | GPT-4o mini | OpenRouter |
| Visão (imagens) | GPT-4o mini | OpenRouter |
| Transcrição de áudio | Whisper-1 | OpenAI direto |
| Embeddings RAG | text-embedding-ada-002 | OpenAI direto |

### Outros modelos compatíveis (via OpenRouter)

| Modelo | OpenRouter ID | Quando usar |
|--------|-------------|-------------|
| GPT-4o | `openai/gpt-4o` | Agente SQL com queries complexas |
| GPT-4.1 | `openai/gpt-4.1` | Melhor custo/benefício OpenAI |
| Claude Sonnet 4.5 | `anthropic/claude-sonnet-4-5` | Melhor em tool use e ReAct |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` | Alternativa open-source |
| DeepSeek V3 | `deepseek/deepseek-chat` | Opção econômica |
| Gemini 2.0 Flash | `google/gemini-2.0-flash-001` | Rápido, sujeito a rate limit 429 |

> Para trocar de modelo: alterar `ROUTER_MODEL_NAME` no `.env` e `docker compose restart bot`. Sem alterar código.

### Prompt caching

Ativo via header `X-OpenRouter-Cache: 1` em todos os modelos. O system prompt (~3.500 tokens) é cacheado a $0,05/M em vez de $0,20/M — redução de ~75% no custo de input em chamadas repetidas.

---

## Parte 8 — Estimativa de Custos

| Tipo de interação | Modelo | Custo (USD) | Custo (BRL) |
|------------------|--------|------------|------------|
| Router (classificação) | Grok 4.1 Fast | ~$0,0002 | ~R$0,001 |
| Mensagem SQL simples | Grok 4.1 Fast | ~$0,002 | ~R$0,012 |
| Mensagem SQL (histórico 30 msgs) | Grok 4.1 Fast | ~$0,003 | ~R$0,018 |
| Mensagem RAG | Grok 4.1 Fast | ~$0,001 | ~R$0,006 |
| Mensagem geral | GPT-4o mini | ~$0,0003 | ~R$0,002 |
| Thinking message | GPT-4o mini | ~$0,00005 | ~R$0,0003 |
| Excel (cache hit) | Zero LLM | $0,00 | R$0,00 |
| Áudio 30s + agente | Whisper + Grok | ~$0,004 | ~R$0,024 |
| Imagem + agente | Vision + Grok | ~$0,002 | ~R$0,012 |
| Indexação PDF ~5 páginas | OpenAI Embeddings | ~$0,001 (único) | ~R$0,006 |

---

## Parte 9 — Docker e Infraestrutura

### Serviços Docker

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `bot` | 8000 | FastAPI + todos os agentes |
| `evolution_api` | 8080/8081 | Gateway WhatsApp |
| `postgres` | 5432/5433 | Banco da Evolution API |
| `redis` | 6379/6380 | Cache, histórico e buffer |

Todos têm health checks. `bot` e `evolution_api` só sobem após Redis e Postgres estarem prontos.

**Bases externas (fora do Docker):**
- Dremio — dados de vendas
- MySQL — dados de compras

### Comandos principais

```bash
# Subir tudo (build completo — necessário após mudar código Python)
docker compose up --build -d

# Restart sem rebuild (após mudar .env ou adicionar PDFs)
docker compose restart bot

# Ver logs em tempo real
docker compose logs -f bot

# Rodar testes localmente (fora do Docker)
pytest tests/
pytest tests/ -v
pytest tests/test_resumo_tool.py -v

# Zerar índice RAG completo
docker compose down && rm -rf ./vectorstore && docker compose up -d

# Limpar cache via API
curl -X POST http://localhost:8000/limpar_cache -H "x-api-key: SUA_KEY"

# Reindexar documentos via API
curl -X POST http://localhost:8000/reindexar -H "x-api-key: SUA_KEY"

# Ver métricas
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

---

## Parte 10 — Como adicionar funcionalidades

### Nova ferramenta SQL (nova tabela Dremio)

1. **`src/tools/dremio_tools.py`** — adicionar classe:
```python
class DremioNovaQueryTool(BaseTool):
    name: str = "nome_da_ferramenta"
    description: str = (
        "QUANDO USAR: ...\n"
        "PALAVRAS-CHAVE: ...\n"
        "Tabela: views.\"ANALYTICS\".\"fNovaTabela\"\n"
        "Colunas disponíveis: coluna1 (TEXT, ...), coluna2 (FLOAT, ...)."
    )
    def _run(self, query: str) -> str:
        return _run_dremio_query("nova", query)
    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
```

2. **`src/chains.py`** — importar e adicionar em `_make_sql_executor()`:
```python
tools = [..., DremioNovaQueryTool()]
```

3. **`src/prompts.py`** — adicionar regra explicando quando usar.

### Nova casa no mapeamento

Editar apenas `src/tools/fantasia_abreviacao.py`:
```python
"NOVA_ABREV": "NOME FANTASIA EXATO NO BANCO",
```

### Novo documento no RAG

1. Colocar PDF/TXT em `rag_files/`
2. Enviar `/reindexar` no WhatsApp (admin) ou chamar o endpoint

---

## Parte 11 — Testes

Os testes rodam **fora do Docker**, com todas as dependências pesadas mockadas no `conftest.py`.

**Cobertura atual:**

| Arquivo | O que testa |
|---------|------------|
| `test_access_control.py` | CRUD de usuários no SQLite |
| `test_app.py` | Comandos admin (autorizar, atualizar, histórico, etc.) |
| `test_chains.py` | Completar datas, classificar intenção, saudações |
| `test_config.py` | Parse de SEED_USERS |
| `test_evolution_api.py` | Split de mensagem, retry, cap de chunks |
| `test_message_buffer.py` | Debounce, cancelamento, criação de tasks |
| `test_resumo_tool.py` | Validação de datas, build_where, execução paralela |
| `test_utils.py` | strip_markdown, extract_json, fmt_brl, fmt_int_br, format_df |

```bash
pytest tests/          # todos
pytest tests/ -v       # verbose (nome de cada teste)
pytest tests/ -q       # quiet (só o resultado final)
```
