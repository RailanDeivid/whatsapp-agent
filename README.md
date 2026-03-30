![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![Grok](https://img.shields.io/badge/xAI-Grok--4.1--fast-000000?logo=x&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-connector-4479A1?logo=mysql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-FF6B35?logo=databricks&logoColor=white)
![WhatsApp](https://img.shields.io/badge/WhatsApp-Evolution%20API-25D366?logo=whatsapp&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)

# whatsapp-agent

Assistente inteligente integrado ao WhatsApp com arquitetura **multi-agente**: um **Router LLM** classifica cada pergunta e roteia para o **Agente SQL** (Grok 4.1 Fast via OpenRouter + Dremio/MySQL, para dados de vendas e compras), o **Agente RAG** (Grok 4.1 Fast via OpenRouter + Chroma, para documentos internos), ou responde diretamente via **LLM** para saudações e perguntas gerais. Respostas gerais e mensagens de espera usam o **Gemini 2.0 Flash** (modelo fallback econômico). Áudios são transcritos via **OpenAI Whisper** antes de chegarem aos agentes.

## Índice

- [Demonstração](#perguntas-sobre-compras)
- [Fluxo Completo](#fluxo-completo--do-whatsapp-à-resposta)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Arquitetura Multi-Agente](#arquitetura-multi-agente)
- [Serviços Docker](#serviços-docker)
- [Controle de Acesso](#controle-de-acesso)
- [Gerenciamento de Documentos (RAG)](#gerenciamento-de-documentos-rag)
- [Endpoints HTTP](#endpoints-http)
- [Ferramentas dos agentes](#ferramentas-dos-agentes)
- [Configuração (.env)](#configuração-env)
- [Subir o projeto](#subir-o-projeto)
- [Personalidade e regras dos agentes](#personalidade-e-regras-dos-agentes)
- [Modelos compatíveis](#modelos-compatíveis)
- [Custo por interação (estimativa)](#custo-por-interação-estimativa)

---

## Perguntas sobre Compras
### Interação por Texto

<img src="image/image-1.png" width="800" alt="Diagrama do fluxo">
<img src="image/image-2.png" width="800" alt="Diagrama do fluxo">
<img src="image/image-3.png" width="800" alt="Diagrama do fluxo">

### Interação por Áudio
<img src="image/image.png" width="800" alt="Diagrama do fluxo">

## Perguntas sobre Vendas
### Interação por Texto
<img src="image/image-4.png" width="800" alt="Diagrama do fluxo">
<img src="image/image-6.png" width="800" alt="Diagrama do fluxo">
<img src="image/image-7.png" width="800" alt="Diagrama do fluxo">

### Interação por Áudio
<img src="image/image-5.png" width="800" alt="Diagrama do fluxo">

## Perguntas sobre Políticas\Regras
<img src="image/image-8.png" width="800" alt="Diagrama do fluxo">

---

## Fluxo Completo — Do WhatsApp à Resposta

<p align="center">
  <img src="src/docs/architecture.svg" width="100%" />
</p>

---

## Estrutura do projeto

```
whatsapp-agent/
├── src/
│   ├── app.py                      # FastAPI — /webhook, /health, /metrics, /limpar_cache, /reindexar + comandos admin
│   ├── access_control.py           # Controle de acesso — SQLite (autorizar, bloquear, remover)
│   ├── chains.py                   # Multi-agente: Router + Agente SQL + Agente RAG + fallback de modelo
│   ├── config.py                   # Leitura das variáveis de ambiente (.env)
│   ├── memory.py                   # Histórico de conversa via Redis (TTL 24h)
│   ├── message_buffer.py           # Buffer de mensagens com debounce + indicador de digitando + reações
│   ├── prompts.py                  # Prompts: ReAct SQL, ReAct RAG, Router, Geral
│   ├── vectorstore.py              # RAG: indexação de PDFs/TXTs via Chroma + OpenAI Embeddings
│   ├── docs/
│   │   └── architecture.svg        # Diagrama do fluxo completo
│   ├── connectors/
│   │   ├── dremio.py               # Conector REST API Dremio → DataFrame
│   │   └── mysql.py                # Conector MySQL → DataFrame (lazy pool)
│   ├── tools/
│   │   ├── dremio_tools.py         # Tools: consultar_vendas, consultar_delivery, consultar_formas_pagamento, consultar_estornos, consultar_metas
│   │   ├── mysql_tools.py          # Tool: consultar_compras
│   │   ├── chart_tool.py           # Tool: gerar_grafico — gráficos PNG via matplotlib/seaborn
│   │   ├── excel_tool.py           # Tool: exportar_excel — planilha .xlsx via pandas/openpyxl
│   │   ├── rag_tool.py             # Tool: consultar_documentos (Chroma)
│   │   ├── utils.py                # strip_markdown — remove blocos sql do output do agente
│   │   └── fantasia_abreviacao.py  # Mapeamento abreviação → nome fantasia do estabelecimento
│   └── integrations/
│       ├── evolution_api.py        # Envio de mensagem/mídia + presence + reações via Evolution API
│       └── transcribe.py           # Transcrição de áudio via OpenAI Whisper
├── data/                           # Banco SQLite de controle de acesso (data/access.db)
├── rag_files/                      # PDFs e TXTs para indexação (apagados após indexar)
├── vectorstore/                    # Índice Chroma gerado automaticamente
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Arquitetura Multi-Agente

```
mensagem → route_and_invoke()
                │
         [Router LLM]         ← classifica a intenção: sql / docs / ambos / geral
         Grok 4.1 Fast (xAI) via OpenRouter  ·  prompt cache ativo
                │
     ┌──────────┼──────────┬──────────┐
   "sql"      "docs"    "ambos"    "geral"
     │           │           │           │
[Agente SQL] [Agente RAG] [Agente SQL] [LLM direto]
Grok 4.1     Grok 4.1     +            Gemini 2.0 Flash
Fast (xAI)   Fast (xAI)   [Agente RAG] sem ferramentas
Dremio+MySQL Chroma       (paralelo — ThreadPoolExecutor)
```

| Rota | Quando aciona | Modelo | Ferramentas |
|---|---|---|---|
| `sql` | Vendas, faturamento, delivery, formas de pagamento, estornos, metas, compras, ticket médio | Grok 4.1 Fast | `consultar_vendas`, `consultar_delivery`, `consultar_formas_pagamento`, `consultar_estornos`, `consultar_metas`, `consultar_cortesias` (Dremio) + `consultar_compras` (MySQL) + `gerar_grafico` + `exportar_excel` |
| `docs` | Políticas, organograma, contatos, emails, ramais | Grok 4.1 Fast | `consultar_documentos` (Chroma) |
| `ambos` | Pergunta envolve dados numéricos E documentos | Grok 4.1 Fast | Agente SQL + Agente RAG **em paralelo** |
| `geral` | Saudações, agradecimentos, fora do escopo, perguntas conceituais | Gemini 2.0 Flash | Nenhuma — LLM direto (sem ReAct) |

**Fallback de modelo:**
- **Respostas gerais e mensagens de espera:** sempre usam `FALLBACK_MODEL_NAME` (Gemini 2.0 Flash) — mais rápido e ~5x mais barato que o modelo principal para tarefas simples.
- **Falha do modelo principal (SQL/RAG):** sistema tenta automaticamente o fallback. Se não configurado, retorna mensagem de erro.
- **Excel fast-path:** se o usuário pedir Excel após uma consulta, o sistema usa o DataFrame já em cache — sem rodar SQL novamente.

---

## Serviços Docker

| Serviço | Imagem | Porta | Função |
|---|---|---|---|
| `bot` | build local | 8000 | FastAPI + Agentes IA |
| `evolution_api` | evoapicloud/evolution-api:latest | 8080 | Gateway WhatsApp |
| `postgres` | postgres:15 | 5432 | Banco de dados da Evolution API |
| `redis` | redis:7 | 6379 | Buffer de mensagens + histórico de conversa |

Todos os serviços possuem **health checks**. O `bot` e a `evolution-api` só sobem após Redis e Postgres estarem prontos.

**Bases de dados externas** (não sobem no Docker):

| Banco | Função |
|---|---|
| Dremio | Dados de vendas — views `fSales`, `fSalesDelivery`, `fFormasPagamento`, `fEstornos`, `dMetas_Casas` |
| MySQL | Dados de compras — tabela `tabela_compras` |

---

## Controle de Acesso

Apenas usuários autorizados conseguem interagir. Os demais recebem uma mensagem configurável via `UNAUTHORIZED_MESSAGE`. Dois perfis: **usuário comum** e **admin**.

Os usuários são armazenados em SQLite (`SQLITE_PATH`, padrão `data/access.db`), persistido via bind mount.

### Configuração inicial — usuários seed

Defina os administradores iniciais via `SEED_USERS` no `.env`. Inseridos automaticamente no primeiro boot e **nunca sobrescritos**.

```env
# Formato: TELEFONE:NOME:CARGO:CASA:admin  (vírgula para múltiplos)
SEED_USERS=5511999990000:João Silva:Analista:Matriz:admin
```

<img src="image/image-10.png" width="800" alt="Controle de acesso">

### Comandos admin (via WhatsApp)

Apenas admins podem usar os comandos abaixo, exceto `/limpar` que está disponível a todos.

| Comando | Descrição |
|---|---|
| `/autorizar 5511999 ; Nome ; Cargo ; Casa` | Adiciona usuário comum |
| `/autorizar 5511999 ; Nome ; Cargo ; Casa ; admin` | Adiciona usuário admin |
| `/bloquear 5511999` | Desativa o acesso (sem apagar o registro) |
| `/desbloquear 5511999` | Reativa usuário bloqueado |
| `/remover 5511999` | Remove permanentemente |
| `/atualizar 5511999 ; 5511888` | Atualiza o número de telefone de um usuário |
| `/usuarios` | Lista usuários ativos e bloqueados |
| `/usuarios admin` | Lista administradores |
| `/historico 5511999` | Exibe o histórico de conversa de um usuário (últimas 72h) |
| `/limpar_usuario 5511999` | Apaga o histórico de conversa de um usuário específico |
| `/reindexar` | Indexa novos arquivos de `rag_files/` sem reiniciar |
| `/limpar` | Apaga o histórico de conversa de todos os usuários |
| `/ajuda` | Exibe lista de comandos |

> Se o número já existir (mesmo bloqueado), `/autorizar` **reativa e atualiza** os dados.

---

## Gerenciamento de Documentos (RAG)

Coloque PDFs ou TXTs na pasta `rag_files/` e o Agente RAG passa a responder perguntas sobre eles.

**Fluxo de indexação:**
1. Coloque o arquivo em `rag_files/`
2. Envie `/reindexar` no WhatsApp (admin) **ou** `docker compose restart bot`
3. O bot extrai o texto, divide em chunks (1000 chars / sobreposição 200), gera embeddings via OpenAI, salva no Chroma e **apaga o arquivo original**

```bash
# Sem reiniciar (recomendado)
/reindexar

# Via endpoint HTTP
curl -X POST http://localhost:8000/reindexar -H "x-api-key: SUA_API_KEY"

# Zerar todo o índice
docker compose down && rm -rf ./vectorstore && docker compose up -d
```

> O custo de embedding (`text-embedding-ada-002`) ocorre apenas na indexação, não nas consultas.

---

## Endpoints HTTP

| Método | Endpoint | Auth | Descrição |
|---|---|---|---|
| `GET` | `/health` | — | Status do servidor |
| `GET` | `/metrics` | — | Métricas de uso (requests, cache, latência, erros) |
| `POST` | `/webhook` | Evolution API | Recebe mensagens do WhatsApp |
| `POST` | `/limpar_cache` | `x-api-key` | Remove respostas cacheadas do Redis |
| `POST` | `/reindexar` | `x-api-key` | Indexa novos arquivos de `rag_files/` |

---

## Ferramentas dos agentes

| Tool | Agente | Fonte | Aciona quando |
|---|---|---|---|
| `consultar_vendas` | SQL | Dremio | Faturamento, receita, desempenho de vendas, ticket médio, fluxo |
| `consultar_delivery` | SQL | Dremio | Pedidos delivery, plataformas (iFood, Rappi, etc.) |
| `consultar_formas_pagamento` | SQL | Dremio | Mix de pagamentos, participação por forma (dinheiro, cartão, pix) |
| `consultar_estornos` | SQL | Dremio | Estornos, cancelamentos, devoluções |
| `consultar_metas` | SQL | Dremio | Metas, orçamento, vendas vs meta, fluxo vs meta |
| `consultar_compras` | SQL | MySQL | Pedidos de compra, fornecedores, notas fiscais |
| `exportar_excel` | SQL | Dremio/MySQL | Quando o usuário pede os dados em `.xlsx` |
| `gerar_grafico` | SQL | Dremio/MySQL | Quando o usuário pede visualização gráfica |
| `consultar_documentos` | RAG | Chroma | Políticas, organograma, contatos, ramais, emails, procedimentos |

> A busca do RAG é **semântica** — encontra a informação mesmo que a pergunta use palavras diferentes do documento.

---

## Configuração (.env)

```env
# Evolution API (WhatsApp)
EVOLUTION_API_URL=http://evolution-api:8080
EVOLUTION_INSTANCE_NAME=instace_name
AUTHENTICATION_API_KEY=sua_api_key

# OpenRouter (modelo principal dos agentes)
ROUTER_API_KEY=sk-or-v1-...
ROUTER_BASE_URL=https://openrouter.ai/api/v1
ROUTER_MODEL_NAME=x-ai/grok-4.1-fast
OPENAI_MODEL_TEMPERATURE=0.3

# Whisper (transcrição de áudio)
WHISPER_API_KEY=sk-or-v1-...

# Redis
BOT_REDIS_URI=redis://redis:6379/0
DEBOUNCE_SECONDS=3

# PostgreSQL (Evolution API)
DATABASE_ENABLED=true
DATABASE_PROVIDER=postgresql
DATABASE_CONNECTION_URI=postgresql://postgres:postgres@postgres:5432/evolution?schema=public

# MySQL (externo)
DB_HOST=seu_host_mysql
DB_PORT=3306
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
DB_NAME=seu_banco

# Dremio (externo)
DREMIO_HOST=seu_host:9047
DREMIO_USER=seu_usuario
DREMIO_PASSWORD=sua_senha

# RAG
RAG_FILES_DIR=rag_files
VECTOR_STORE_PATH=vectorstore

# Fallback — respostas gerais, thinking message e erros do modelo principal
FALLBACK_MODEL_NAME=google/gemini-2.0-flash-001

# Histórico de conversa por sessão (número de mensagens)
CONVERSATION_MAX_HISTORY=30

# Controle de acesso
SQLITE_PATH=data/access.db
UNAUTHORIZED_MESSAGE=Olá! Você não está autorizado a usar este assistente. Entre em contato com um administrador.
SEED_USERS=5511999990000:João Silva:TI:Matriz:admin
```

---

## Subir o projeto

```bash
# Primeira vez ou após mudanças no código
docker compose up --build -d

# Reiniciar sem rebuild (após mudanças no .env ou adição de PDFs)
docker compose restart bot

# Ver logs em tempo real
docker compose logs -f bot
```

Os agentes usam **inicialização lazy** — o modelo e as ferramentas são carregados apenas na **primeira mensagem recebida**, não no boot.

---

## Personalidade e regras dos agentes

Comportamento definido em [src/prompts.py](src/prompts.py).

| Agente | Regras principais |
|---|---|
| **SQL** (`react_prompt`) | Nunca revela tabelas/schemas. Sempre consulta as tools. Datas sem ano completadas com o ano corrente. SSS calculado via CTE sem perguntar ao usuário. Vendas por vertical retornam casa a casa + total consolidado. Vs meta somente quando o usuário pedir explicitamente. Histórico: 30 mensagens por sessão. |
| **RAG** (`rag_prompt`) | Responde somente com base nos documentos indexados. Se não encontrar, informa claramente. Nunca inventa informações. |
| **Geral** (`general_prompt`) | Gemini 2.0 Flash, sem ReAct. Saudações, perguntas conceituais, fora do escopo. Mais rápido e barato. |
| **Router** (`router_prompt`) | Grok 4.1 Fast. Classifica em `sql`, `docs`, `ambos` ou `geral`. Fallback para `sql` em caso de resposta inválida. |

> Ambos os agentes compartilham o mesmo histórico Redis por sessão (TTL 24h). O usuário pode alternar entre perguntas de dados e documentos livremente.

---

## Modelos compatíveis

### Agentes (chat / ReAct) — via OpenRouter

| Modelo | Uso | Preço (input/output por 1M tokens) |
|---|---|---|
| `x-ai/grok-4.1-fast` | Modelo principal — SQL Agent, RAG Agent, Router | $0,20 / $0,50 |
| `google/gemini-2.0-flash-001` | Fallback — respostas gerais e mensagens de espera | $0,10 / $0,40 |
| `x-ai/grok-4` | Raciocínio profundo — não recomendado: reasoning obrigatório e 15x mais caro | $3,00 / $15,00 |

> O OpenRouter permite trocar o modelo sem alterar código via `ROUTER_MODEL_NAME` e `FALLBACK_MODEL_NAME` no `.env`. Cache de prompt ativo via header `X-OpenRouter-Cache` — reduz custo do prompt do sistema de $0,20 para $0,05/M tokens.

### Embeddings e Transcrição

| Modelo | Uso | Custo |
|---|---|---|
| `text-embedding-ada-002` | Indexação e busca RAG (OpenAI) | ~$0,0001/1k tokens |
| `whisper-1` | Transcrição de áudio | ~$0,006/min |

---

## Custo por interação (estimativa)

| Tipo de interação | Modelo usado | Custo (USD) | Custo (BRL) |
|---|---|---|---|
| Router (classificação) | Grok 4.1 Fast | ~$0,0002 | ~R$0,001 |
| Mensagem SQL simples | Grok 4.1 Fast | ~$0,002 | ~R$0,012 |
| Mensagem SQL (histórico 30 msgs) | Grok 4.1 Fast | ~$0,003 | ~R$0,018 |
| Mensagem RAG | Grok 4.1 Fast | ~$0,001 | ~R$0,006 |
| Mensagem geral / conceitual | Gemini 2.0 Flash | ~$0,0003 | ~R$0,002 |
| Thinking message (espera) | Gemini 2.0 Flash | ~$0,00005 | ~R$0,0003 |
| Excel (dados já em cache) | Zero LLM | $0,00 | R$0,00 |
| Áudio 30s + agente | Whisper + Grok | ~$0,004 | ~R$0,024 |
| Indexação PDF (~5 páginas) | OpenAI Embeddings | ~$0,001 (uma vez) | ~R$0,006 |

> Prompt cache ativo — o prompt do sistema (~3.500 tokens) é cacheado a $0,05/M em vez de $0,20/M, reduzindo ~25% do custo de input em todas as chamadas. Consulte [openrouter.ai/models](https://openrouter.ai/models) para preços atualizados.
