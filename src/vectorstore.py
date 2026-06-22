import logging
import os
import threading

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import RAG_FILES_DIR, VECTOR_STORE_PATH, WHISPER_API_KEY

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
_reload_lock = threading.Lock()


def load_documents():
    processed_dir = os.path.join(RAG_FILES_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    docs = []
    files = [
        os.path.join(RAG_FILES_DIR, f)
        for f in os.listdir(RAG_FILES_DIR)
        if f.endswith('.pdf') or f.endswith('.txt')
    ]
    for file in files:
        basename = os.path.basename(file)
        try:
            loader = PyPDFLoader(file) if file.endswith('.pdf') else TextLoader(file, encoding="utf-8")
            loaded = loader.load()
            if not loaded:
                logger.warning("[vectorstore] Arquivo sem conteudo extraido, mantendo em rag_files/: %s", basename)
                continue
            docs.extend(loaded)
            dest = os.path.join(processed_dir, basename)
            os.rename(file, dest)
            logger.info("[vectorstore] Arquivo indexado e movido para processed/: %s (%d paginas)", basename, len(loaded))
        except Exception as e:
            logger.error("[vectorstore] Erro ao processar '%s' — arquivo mantido para reprocessamento: %s", basename, e)
    return docs


def get_vectorstore():
    docs = load_documents()
    if docs:
        splits = _splitter.split_documents(docs)
        return Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(api_key=WHISPER_API_KEY),
            persist_directory=VECTOR_STORE_PATH,
        )
    # Abre índice existente se o diretório já tiver dados
    if os.path.isdir(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return Chroma(
            embedding_function=OpenAIEmbeddings(api_key=WHISPER_API_KEY),
            persist_directory=VECTOR_STORE_PATH,
        )
    return None


def reload_vectorstore() -> tuple[bool, str]:
    """
    Indexa novos arquivos de RAG_FILES_DIR sem reiniciar o servidor.
    Retorna (sucesso, mensagem).
    """
    from src.tools.rag_tool import invalidate_vectorstore

    with _reload_lock:
        files = [
            f for f in os.listdir(RAG_FILES_DIR)
            if f.endswith('.pdf') or f.endswith('.txt')
        ]
        if not files:
            return False, "Nenhum arquivo novo encontrado em rag_files."

        logger.info("[reindex] Indexando %d arquivo(s): %s", len(files), files)
        docs = load_documents()
        if not docs:
            return False, "Nenhum conteudo extraido dos arquivos."

        splits = _splitter.split_documents(docs)

        # Adiciona ao índice existente ou cria novo
        if os.path.isdir(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vs = Chroma(
                embedding_function=OpenAIEmbeddings(api_key=WHISPER_API_KEY),
                persist_directory=VECTOR_STORE_PATH,
            )
            vs.add_documents(splits)
        else:
            Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(api_key=WHISPER_API_KEY),
                persist_directory=VECTOR_STORE_PATH,
            )

        invalidate_vectorstore()
        logger.info("[reindex] %d chunks adicionados ao vectorstore.", len(splits))
        return True, f"{len(files)} arquivo(s) indexado(s) com {len(splits)} chunks."