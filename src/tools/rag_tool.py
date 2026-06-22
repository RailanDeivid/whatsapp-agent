import asyncio
import logging
import threading

from langchain.tools import BaseTool

from src.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

_vs_lock = threading.Lock()
_vs_loaded = False
_vectorstore = None

_NO_DOCS_MSG = (
    "Nenhum documento foi indexado ainda. "
    "Adicione PDFs ou TXTs à pasta rag_files e reinicie o bot."
)


def _get_vs():
    global _vectorstore, _vs_loaded
    if not _vs_loaded:
        with _vs_lock:
            if not _vs_loaded:
                logger.info("Carregando vectorstore...")
                _vectorstore = get_vectorstore()
                _vs_loaded = True
                if _vectorstore is None:
                    logger.warning("Nenhum documento indexado — vectorstore vazio.")
                else:
                    logger.info("Vectorstore pronto.")
    return _vectorstore


def invalidate_vectorstore() -> None:
    """Força recarga do vectorstore na próxima consulta (chamado após reindexação)."""
    global _vectorstore, _vs_loaded
    with _vs_lock:
        _vectorstore = None
        _vs_loaded = False
    logger.info("Cache do vectorstore invalidado — sera recarregado na proxima consulta.")


class RAGDocumentQueryTool(BaseTool):
    name: str = "consultar_documentos"
    description: str = (
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre documentos internos, "
        "politicas, procedimentos, organograma, contatos, emails, ramais ou informacoes institucionais. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: politica, procedimento, organograma, contato, "
        "email, ramal, quem procurar, responsavel, manual, regulamento, norma, regra interna, "
        "estrutura organizacional, setor, departamento, quem cuida, quem e responsavel. "
        "NAO use para dados de vendas, faturamento, compras ou fornecedores. "
        "NUNCA invente informacoes — se nao encontrar nos documentos, informe que nao tem acesso. "
        "Input: pergunta em linguagem natural."
    )

    def _run(self, query: str) -> str:
        logger.info("Buscando nos documentos: %.80s", query)
        vs = _get_vs()
        if vs is None:
            return _NO_DOCS_MSG
        try:
            docs = vs.similarity_search(query, k=5)
            if not docs:
                return "Nenhum documento relevante encontrado para essa pergunta."
            return "\n\n".join(
                f"[Trecho {i + 1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
        except Exception as e:
            logger.error("ERRO RAG: %s: %s", type(e).__name__, e)
            return f"Erro ao consultar documentos: {str(e)}"

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
