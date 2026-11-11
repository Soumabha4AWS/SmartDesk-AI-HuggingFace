from pathlib import Path
from typing import Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_rag_pipeline(
    documents_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
):
    """Build core RAG components: docs -> chunks -> embeddings -> vectorstore -> retriever.

    Returns a dict with: vectorstore, retriever, format_docs.
    """
    # Load documents
    document_files = [
        "hr-policies.md", "it-support.md",
        "onboarding.md", "payroll.md",
    ]
    all_documents = []
    for filename in document_files:
        content = (documents_dir / filename).read_text(encoding="utf-8")
        all_documents.append(Document(page_content=content, metadata={"source": filename}))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_documents)

    # Embed locally with a sentence-transformers model and store in Chroma.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name=collection_name
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', '')}]\n{doc.page_content}"
            for doc in docs
        )

    return {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "format_docs": format_docs,
    }
