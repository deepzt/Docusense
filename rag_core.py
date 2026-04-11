"""Core RAG pipeline: ingest → embed → index → retrieve → rerank → generate."""

import os
import re
import time
import uuid
import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any, Iterator
import csv
from datetime import datetime

logger = logging.getLogger(__name__)

import pandas as pd
from dotenv import load_dotenv

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ── Module-level singletons ───────────────────────────────────────────────────

embeddings_model: Optional[HuggingFaceEmbeddings] = None
embedding_model_name = "BAAI/bge-base-en-v1.5"

reranker_model: Optional[CrossEncoder] = None
reranker_model_name = "BAAI/bge-reranker-base"

# Rolling history of best reranker scores — used for adaptive confidence threshold
_session_best_scores: List[float] = []
_MAX_SCORE_HISTORY = 200

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
POPPLER_PATH  = os.getenv("POPPLER_PATH")

if OCR_AVAILABLE and TESSERACT_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # type: ignore[attr-defined]
    except Exception:
        pass

# ── Device / model helpers ────────────────────────────────────────────────────

def _get_torch_device() -> str:
    if not _TORCH_AVAILABLE:
        return "cpu"
    try:
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                if free_bytes > 2 * 1024 ** 3:
                    return "cuda"
            except Exception:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embeddings_model() -> HuggingFaceEmbeddings:
    global embeddings_model
    if embeddings_model is None:
        device = _get_torch_device()
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )
    return embeddings_model


def get_reranker_model() -> CrossEncoder:
    global reranker_model
    if reranker_model is None:
        device = _get_torch_device()
        reranker_model = CrossEncoder(reranker_model_name, device=device)
    return reranker_model


# ── Heading / section detection ───────────────────────────────────────────────

def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^(\d+(\.\d+)*[\.)]?\s+).+", stripped):
        return True
    if 3 <= len(stripped) <= 80 and stripped.isupper():
        return True
    if 3 <= len(stripped) <= 80 and stripped[0].isupper() and not stripped.endswith("."):
        if len(stripped.split()) <= 10:
            return True
    return False


def _split_text_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    lines = text.splitlines()
    sections: List[Tuple[Optional[str], str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if _looks_like_heading(stripped):
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_title, section_text))
                current_lines = []
            current_title = stripped
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_title, section_text))

    return sections if sections else [(None, text)]


# ── Document creation ─────────────────────────────────────────────────────────

def create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    documents: List[Document] = []
    columns = df.columns.tolist()

    summary_parts = [
        f"Table summary: {len(df)} rows, {len(columns)} columns.",
        f"Columns: {', '.join(columns)}",
    ]
    try:
        desc = df.describe(include="all")
        for col in desc.columns:
            col_stats = [f"{stat}={val}" for stat, val in desc[col].items() if pd.notna(val)]
            if col_stats:
                summary_parts.append(f"  {col}: {', '.join(col_stats)}")
        unique_counts = df.nunique()
        summary_parts.append(
            "Unique value counts: " + ", ".join(f"{c}={unique_counts[c]}" for c in columns)
        )
    except Exception:
        pass

    documents.append(Document(
        page_content="\n".join(summary_parts),
        metadata={"source_type": "table_summary", "columns": ", ".join(columns)},
    ))

    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
        base_text = " | ".join(parts)
        summary = f"Row {int(idx)} from table with columns: {', '.join(columns)}."
        enriched_text = (
            f"Source type: table\n"
            f"Row index: {int(idx)}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Chunk summary: {summary}\n\n"
            f"Row data: {base_text}"
        )
        documents.append(Document(
            page_content=enriched_text,
            metadata={
                "row_index": int(idx),
                "columns": ", ".join(columns),
                "source_type": "table",
                "summary": summary,
                "chunk_id": int(idx),
            },
        ))
    return documents


def create_documents_from_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    page_docs = loader.load()

    if not page_docs or all(not (doc.page_content or "").strip() for doc in page_docs):
        if not OCR_AVAILABLE:
            raise ValueError(
                "This PDF appears to contain no selectable text. OCR is not configured. "
                "Please upload a text-based PDF or configure OCR (TESSERACT_CMD, POPPLER_PATH)."
            )
        images = convert_from_path(path, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(path)
        ocr_docs: List[Document] = []
        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={"source_type": "pdf_ocr", "source_file": os.path.basename(path), "page": page_num},
                ))
        if not ocr_docs:
            raise ValueError("Unable to extract any text from this PDF, even with OCR.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250, separators=["\n\n", "\n", ". ", " ", ""])
        chunk_docs = splitter.split_documents(ocr_docs)
    else:
        base_name = os.path.basename(path)
        for doc in page_docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.setdefault("source_type", "pdf")
            doc.metadata.setdefault("source_file", base_name)

        section_docs: List[Document] = []
        for doc in page_docs:
            sections = _split_text_into_sections(doc.page_content or "")
            for sec_idx, (title, sec_text) in enumerate(sections):
                meta = dict(doc.metadata or {})
                if title:
                    meta["section_title"] = title
                meta["section_index"] = sec_idx
                section_docs.append(Document(page_content=sec_text, metadata=meta))

        # Semantic chunking: split at topic-shift boundaries, not fixed char windows
        chunk_docs = []
        for sec_doc in section_docs:
            for chunk_text in _semantic_chunk_text(sec_doc.page_content, target_size=900):
                chunk_docs.append(Document(
                    page_content=chunk_text,
                    metadata=dict(sec_doc.metadata),
                ))

    base_name = os.path.basename(path)
    for idx, doc in enumerate(chunk_docs):
        doc.metadata.setdefault("chunk_id", idx)
        page = doc.metadata.get("page")
        source_type = doc.metadata.get("source_type", "pdf")
        summary = (
            f"Chunk {idx} from PDF '{base_name}' on page {page}"
            if page is not None else f"Chunk {idx} from PDF '{base_name}'"
        )
        doc.page_content = (
            f"Source file: {base_name}\n"
            f"Source type: {source_type}\n"
            f"Page: {page}\n"
            f"Chunk summary: {summary}\n\n"
            f"Content:\n{doc.page_content}"
        )
        doc.metadata.setdefault("source_file", base_name)
        doc.metadata["summary"] = summary

    return chunk_docs


def create_documents_from_txt(path: str) -> List[Document]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        raise ValueError(f"Error reading text file: {e}")

    if not text or not text.strip():
        return []

    sections = _split_text_into_sections(text)

    documents: List[Document] = []
    base_name = os.path.basename(path)
    for sec_idx, (title, sec_text) in enumerate(sections):
        chunks = _semantic_chunk_text(sec_text, target_size=700)
        for idx, chunk in enumerate(chunks):
            global_chunk_id = sec_idx * 100000 + idx
            summary = f"Chunk {global_chunk_id} from text/log file '{base_name}'."
            documents.append(Document(
                page_content=(
                    f"Source file: {base_name}\n"
                    f"Source type: text/log\n"
                    f"Section index: {sec_idx}\n"
                    f"Section title: {title or ''}\n"
                    f"Chunk id: {global_chunk_id}\n"
                    f"Chunk summary: {summary}\n\n"
                    f"Content:\n{chunk}"
                ),
                metadata={
                    "source_type": "text",
                    "source_file": base_name,
                    "section_index": sec_idx,
                    "section_title": title or "",
                    "chunk_id": global_chunk_id,
                    "summary": summary,
                },
            ))
    return documents


# ── Retrieval models ──────────────────────────────────────────────────────────

def get_openai_model() -> Tuple[Optional[ChatOpenAI], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    # Prefer gpt-4o-mini; instantiate without a test call so startup never blocks
    # on network latency. Bad keys surface on the first real query instead.
    name = "gpt-4o-mini"
    try:
        model = ChatOpenAI(model=name, temperature=0.2, api_key=api_key)
        return model, name
    except Exception:
        return None, None


def get_ollama_model() -> Tuple[Optional[ChatOllama], Optional[str]]:
    for name in ["llama3.1", "gemma"]:
        try:
            model = ChatOllama(model=name, temperature=0.4, validate_model_on_init=True)
            _ = model.invoke("Hi")
            return model, name
        except Exception:
            continue
    return None, None


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_results(
    question: str,
    results: List[Tuple[Document, float]],
    top_k: int = 3,
) -> List[Tuple[Document, float]]:
    if not results:
        return results
    reranker = get_reranker_model()
    pairs = [(question, doc.page_content) for doc, _ in results]
    scores = reranker.predict(pairs)
    scored = list(zip(results, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)
    return [(doc, float(rerank_score)) for (doc, _), rerank_score in scored[:top_k]]


# ── LLM model cache ───────────────────────────────────────────────────────────

_model_temp_cache: Dict[Tuple[str, str, float], Any] = {}


def _get_model_with_temperature(base_model: Any, model_name: Optional[str], temperature: float) -> Any:
    if base_model is None or model_name is None:
        return base_model
    key = (type(base_model).__name__, model_name, temperature)
    if key not in _model_temp_cache:
        if isinstance(base_model, ChatOpenAI):
            _model_temp_cache[key] = ChatOpenAI(model=model_name, temperature=temperature)
        elif isinstance(base_model, ChatOllama):
            _model_temp_cache[key] = ChatOllama(model=model_name, temperature=temperature)
        else:
            return base_model
    return _model_temp_cache[key]


def _select_model(
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
    temperature: Optional[float],
) -> Tuple[Any, Optional[str]]:
    if openai_model is not None:
        model, name = openai_model, openai_model_name or "OpenAI"
    elif ollama_model is not None:
        model, name = ollama_model, ollama_model_name or "Ollama"
    else:
        return None, None
    if temperature is not None:
        model = _get_model_with_temperature(model, name, temperature)
    return model, name


# ── Hybrid retrieval helpers ──────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    results_lists: List[List[Tuple[Document, float]]],
    rrf_k: int = 60,
) -> List[Tuple[Document, float]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for results in results_lists:
        for rank, (doc, _) in enumerate(results):
            key = doc.page_content[:300]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs[key] = doc
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs[k], score) for k, score in merged]


def _bm25_search(
    bm25: BM25Okapi,
    docs: List[Document],
    query: str,
    k: int,
) -> List[Tuple[Document, float]]:
    """BM25 search with normalized scores instead of rank-only 0.0 placeholders."""
    import numpy as np
    tokens = query.lower().split()
    raw_scores = bm25.get_scores(tokens)
    max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
    normalised = raw_scores / max_score  # scale to [0, 1]
    top_idx = raw_scores.argsort()[::-1][:k]
    return [(docs[i], float(normalised[i])) for i in top_idx if raw_scores[i] > 0]


# ── Improvement 1 — Parent store (small-to-big retrieval) ────────────────────

def _build_parent_store(documents: List[Document]) -> Dict[str, str]:
    """Group child chunks by their page/section and return a parent_id → full_text map.

    Child chunks are indexed for precision; at answer time the full parent window
    (the original page or section) is sent to the LLM for richer context.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)

    for doc in documents:
        src_file   = doc.metadata.get("source_file", "")
        page       = doc.metadata.get("page")
        sec_idx    = doc.metadata.get("section_index")
        src_type   = doc.metadata.get("source_type", "")

        # Table rows are atomic — each row is its own parent
        if src_type in ("table", "table_summary"):
            parent_id = f"tbl_{src_file}_{doc.metadata.get('row_index', id(doc))}"
        elif page is not None:
            parent_id = f"{src_file}_p{page}"
        elif sec_idx is not None:
            parent_id = f"{src_file}_s{sec_idx}"
        else:
            parent_id = f"{src_file}_doc"

        doc.metadata["parent_id"] = parent_id
        cleaned = _clean_chunk(doc.page_content)
        buckets[parent_id].append(cleaned)

    return {pid: "\n\n".join(chunks) for pid, chunks in buckets.items()}


# ── Improvement 2 — Chunk deduplication ──────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate_chunks(
    kept: List[Tuple[Document, float]],
    threshold: float = 0.82,
) -> List[Tuple[Document, float]]:
    """Remove near-duplicate chunks (Jaccard ≥ threshold) keeping the higher-scored one."""
    unique: List[Tuple[Document, float]] = []
    for doc, score in kept:
        content = _clean_chunk(doc.page_content)
        is_dup = any(_jaccard(content, _clean_chunk(u.page_content)) >= threshold for u, _ in unique)
        if not is_dup:
            unique.append((doc, score))
    return unique


# ── Improvement 7 — Semantic chunking ────────────────────────────────────────

def _semantic_chunk_text(
    text: str,
    target_size: int = 800,
    min_sentences: int = 3,
    breakpoint_threshold: float = 0.35,
) -> List[str]:
    """Split text at semantic topic boundaries using embedding similarity.

    Consecutive sentences whose cosine similarity drops below *breakpoint_threshold*
    are treated as a topic shift. Resulting chunks respect *target_size* (characters)
    and always contain at least *min_sentences* sentences.

    Falls back to a single chunk for very short texts.
    """
    # Split into sentences on common terminators
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]

    if len(sentences) <= min_sentences:
        return [text.strip()] if text.strip() else []

    # Embed all sentences in one batch — embedding model is already loaded
    emb_model = get_embeddings_model()
    try:
        embs = emb_model.embed_documents(sentences)
    except Exception:
        # On failure, fall back to a single chunk
        return [text.strip()]

    emb_matrix = np.array(embs, dtype=np.float32)
    # Normalise rows so dot-product == cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
    emb_matrix /= norms

    # Identify sentence indices that start a new topic
    breakpoints: set = set()
    for i in range(1, len(sentences)):
        sim = float(np.dot(emb_matrix[i - 1], emb_matrix[i]))
        if sim < breakpoint_threshold:
            breakpoints.add(i)

    # Group sentences into chunks
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for i, sent in enumerate(sentences):
        current.append(sent)
        current_len += len(sent) + 1  # +1 for the space separator

        at_break   = (i + 1) in breakpoints
        over_limit = current_len >= target_size
        has_min    = len(current) >= min_sentences

        if (at_break or over_limit) and has_min:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        tail = " ".join(current)
        # Merge a short tail into the previous chunk to avoid tiny orphan chunks
        if chunks and current_len < target_size // 3:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)

    return chunks if chunks else [text.strip()]


# ── Improvement 3 — Adaptive confidence threshold ────────────────────────────

def _get_adaptive_threshold() -> float:
    """Return a confidence floor based on session score history.

    Falls back to 0.3 until we have ≥15 data points. After that, uses the
    20th-percentile of observed best-scores so the threshold auto-calibrates
    to the document's score distribution.
    """
    if len(_session_best_scores) < 15:
        return 0.3
    sorted_s = sorted(_session_best_scores)
    p20_idx  = max(0, int(len(sorted_s) * 0.20) - 1)
    return max(0.05, min(sorted_s[p20_idx], 0.4))


# ── Improvement 8 — Faithfulness check ──────────────────────────────────────

def _faithfulness_score(answer: str, context_block: str) -> float:
    """Score how well the answer is grounded in context via the CrossEncoder.

    The reranker is a relevance/entailment model: a high positive score means
    the context *supports* the answer; a large negative score suggests the
    answer introduces facts not present in the context (hallucination signal).
    Returns 0.0 on any error so failures never suppress valid answers.
    """
    try:
        reranker = get_reranker_model()
        scores = reranker.predict([(context_block[:2000], answer[:1200])])
        return float(scores[0])
    except Exception:
        return 0.0


def _append_faithfulness_warning(answer: str, context_block: str) -> str:
    """Append a reliability notice when the faithfulness score is very low."""
    score = _faithfulness_score(answer, context_block)
    if score < -1.5:
        return answer + (
            "\n\n---\n> ⚠ **Reliability notice:** This response may contain "
            "information not directly found in the uploaded document. "
            "Please verify key claims against the source."
        )
    return answer


# ── Improvement 4 — Conditional query rewriting ──────────────────────────────

def _needs_query_rewrite(question: str) -> bool:
    """Return False for short/simple queries that don't benefit from rewriting.

    Rewriting adds a full LLM round-trip (~2–8 s). Very short queries, direct
    keyword lookups, and exact-phrase searches are already retrieval-ready.
    """
    words = question.strip().split()
    if len(words) <= 3:
        return False                   # "What is X?" — already specific
    if re.match(r'^"[^"]+"$', question.strip()):
        return False                   # Quoted exact phrase
    return True


# ── Improvement 9 — HNSW vector index ────────────────────────────────────────

def _build_hnsw_faiss_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> FAISS:
    """Build a FAISS vector store backed by an HNSW index.

    HNSW (Hierarchical Navigable Small World) provides sub-linear query time
    with >99 % recall versus the flat exhaustive index, and speeds up noticeably
    for corpora larger than a few thousand chunks.

    Parameters
    ----------
    M               : edges per node — higher → better recall, more RAM
    ef_construction : build-time search width — higher → better graph quality
    ef_search       : query-time beam width — set after build, trades speed/recall
    """
    import faiss as faiss_lib

    texts      = [d.page_content for d in documents]
    emb_list   = embeddings.embed_documents(texts)
    emb_matrix = np.array(emb_list, dtype=np.float32)
    dim        = emb_matrix.shape[1]

    # Normalise so that L2 distance ≡ cosine distance on unit vectors
    faiss_lib.normalize_L2(emb_matrix)

    # Build the HNSW graph
    index = faiss_lib.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch       = ef_search
    index.add(emb_matrix)

    # Build the LangChain-compatible docstore
    ids                   = [str(uuid.uuid4()) for _ in documents]
    docstore              = InMemoryDocstore({ids[i]: documents[i] for i in range(len(documents))})
    index_to_docstore_id  = {i: ids[i] for i in range(len(documents))}

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


# ── Index building ────────────────────────────────────────────────────────────

def build_vector_store_for_file(
    file_path: str,
) -> Tuple[FAISS, Dict[str, str], str]:
    """Build FAISS index + parent store for a file.

    Returns (faiss_store, parent_store, info_string).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        documents = create_documents_from_dataframe(df)
        source_desc = f"CSV with {len(df)} rows"
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
        documents = create_documents_from_dataframe(df)
        source_desc = f"Excel with {len(df)} rows"
    elif ext == ".pdf":
        documents = create_documents_from_pdf(file_path)
        source_desc = f"PDF with {len(documents)} chunks"
    elif ext == ".txt":
        documents = create_documents_from_txt(file_path)
        source_desc = f"TXT with {len(documents)} chunks"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not documents:
        raise ValueError("No content found in the file to index.")

    # Build parent store BEFORE embedding so parent_id metadata is present
    parent_store = _build_parent_store(documents)

    embeddings  = get_embeddings_model()
    faiss_store = _build_hnsw_faiss_store(documents, embeddings)

    info = (
        f"Built HNSW-FAISS index for {os.path.basename(file_path)} "
        f"from {source_desc} ({len(documents)} chunks, "
        f"{len(parent_store)} parent windows)."
    )
    return faiss_store, parent_store, info


def build_rag_index(
    file_path: str,
) -> Tuple[FAISS, Tuple[BM25Okapi, List[Document]], Dict[str, str], str]:
    """Build FAISS + BM25 + parent store for a file.

    Returns (faiss_store, (bm25_index, bm25_docs), parent_store, info_string).
    """
    faiss_store, parent_store, info = build_vector_store_for_file(file_path)
    all_docs = list(faiss_store.docstore._dict.values())
    tokenized = [doc.page_content.lower().split() for doc in all_docs]
    bm25      = BM25Okapi(tokenized)
    return faiss_store, (bm25, all_docs), parent_store, info


# ── Query rewriting ───────────────────────────────────────────────────────────

def _rewrite_query_with_llm(
    question: str,
    history: Optional[List[Tuple[str, str]]],
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Rewrite the query into DETAILED / EXPANDED / KEYWORDS forms.

    Skipped for short or simple queries to avoid unnecessary LLM latency.
    Returns (detailed, expanded, keywords) — any may be None on skip/failure.
    """
    # Improvement 4: skip rewriting for simple queries
    if not _needs_query_rewrite(question):
        return question, None, None

    selected_model = openai_model or ollama_model
    if selected_model is None:
        return question, None, None

    history = history or []
    recent  = history[-3:]
    history_text = (
        "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent if q and a)
        if recent else "(none)"
    )

    rewrite_prompt = (
        "You are a query rewriting assistant for a retrieval system. Given the user's "
        "current question and recent Q&A history, rewrite the question in three useful "
        "ways to improve retrieval from a vector database.\n\n"
        "STRICT OUTPUT FORMAT (no extra text):\n"
        "DETAILED: <a more detailed, explicit version of the question>\n"
        "EXPANDED: <an expanded version that includes synonyms and related concepts>\n"
        "KEYWORDS: <a compact comma-separated list of key terms and entities>\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Original question: {question}"
    )

    try:
        response = selected_model.invoke(rewrite_prompt)
        text     = getattr(response, "content", "") or ""
        detailed = expanded = keywords = question
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith("DETAILED:"):
                detailed = s[len("DETAILED:"):].strip() or question
            elif s.upper().startswith("EXPANDED:"):
                expanded = s[len("EXPANDED:"):].strip() or question
            elif s.upper().startswith("KEYWORDS:"):
                keywords = s[len("KEYWORDS:"):].strip() or question
        return detailed, expanded, keywords
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)
        return question, None, None


# ── System prompts ────────────────────────────────────────────────────────────

_SUMMARIZATION_PROMPT = """\
You are DocuSense Pro, a document summarization assistant.
Summarize the document using ONLY the information in the CONTEXT below. Do not invent facts.

Write your summary in this structure (skip a section only if the context has no relevant content):

## Overview
Two to four sentences describing the document's purpose, scope, and audience.

## Main Points
A bullet list. One concise bullet per major topic or section found in the context.
Cite each bullet with its source number, e.g. [1].

## Key Details
Important figures, dates, names, decisions, or requirements. Use a bullet list.
Cite each item, e.g. [2].

## Takeaways
Three to five short bullets summarising the most important things to remember.

If the context appears incomplete or covers only part of the document, say so briefly at the end.
"""


def _get_qa_system_prompt() -> str:
    return """\
You are DocuSense Pro, a precise document Q&A assistant.
Answer the user's question using ONLY the information in the CONTEXT below. Never fabricate.

Response guidelines:
- Match length to complexity: simple questions get 1–3 sentences; complex ones get structured paragraphs.
- Cite every factual claim inline using the source numbers provided, e.g. [1] or [2].
- Use **bold** for key terms, bullet lists for enumerable items, and markdown tables for comparisons.
- If the context does not contain enough information to answer, say so clearly instead of guessing.
- Do not pad answers with unnecessary headers or restate the question.
- End with a single **> Key takeaway:** line (one sentence) only when it adds genuine value.
"""


# ── Context assembly ──────────────────────────────────────────────────────────

def _clean_chunk(content: str) -> str:
    """Strip indexing boilerplate, returning only the content text."""
    if "\nContent:\n" in content:
        return content.split("\nContent:\n", 1)[1].strip()
    for line in content.splitlines():
        if line.startswith("Row data:"):
            return line[len("Row data:"):].strip()
    return content.strip()


def _build_context_block(
    kept: List[Tuple[Document, float]],
    is_summarization: bool,
    parent_store: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Format retrieved docs into a numbered context block + sources legend.

    If parent_store is provided, expands each chunk to its full parent window
    (Improvement 1: small-to-big retrieval).
    """
    _PARENT_MAX = 2500   # cap parent window to avoid flooding context
    _CHILD_MAX  = 1800 if is_summarization else 900

    context_parts: List[str] = []
    legend_parts:  List[str] = []

    for i, (doc, _score) in enumerate(kept, 1):
        parent_id = doc.metadata.get("parent_id")

        if parent_store and parent_id and parent_id in parent_store:
            # Use the fuller parent window for the LLM
            text = parent_store[parent_id]
            if len(text) > _PARENT_MAX:
                text = text[:_PARENT_MAX] + " …"
        else:
            # Fall back to the child chunk itself
            text = _clean_chunk(doc.page_content)
            if len(text) > _CHILD_MAX:
                text = text[:_CHILD_MAX] + " …"

        src_type = doc.metadata.get("source_type", "document")
        page     = doc.metadata.get("page")
        src_file = doc.metadata.get("source_file", "")

        if page is not None:
            label = f"{src_file}, page {page}" if src_file else f"page {page}"
        elif src_file:
            label = src_file
        else:
            label = src_type

        context_parts.append(f"[{i}]\n{text}\n")
        legend_parts.append(f"[{i}] {label}")

    context_block  = "\n".join(context_parts)
    sources_legend = "Sources:\n" + "\n".join(legend_parts)
    return context_block, sources_legend


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_rag_event(
    question: str,
    top_scores: List[float],
    kept_k: int,
    threshold: float,
    confidence_ok: bool,
    model_name: Optional[str],
    latency_s: float,
    answer_len: int,
) -> None:
    """Append one row to logs/rag_log.csv with enriched pipeline metrics."""
    try:
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        logs_dir  = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path  = os.path.join(logs_dir, "rag_log.csv")
        _max_log_bytes = 50 * 1024 * 1024
        if os.path.exists(log_path) and os.path.getsize(log_path) >= _max_log_bytes:
            os.replace(log_path, log_path + ".1")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            ts = datetime.utcnow().isoformat()
            writer.writerow([
                ts,
                question,
                "|".join(f"{s:.4f}" for s in top_scores[:3]),
                kept_k,
                f"{threshold:.3f}",
                confidence_ok,
                model_name or "none",
                f"{latency_s:.2f}",
                answer_len,
            ])
    except Exception:
        pass


# ── Core retrieval pipeline ───────────────────────────────────────────────────

def _retrieve_and_build_context(
    vector_store: FAISS,
    question: str,
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
    history: Optional[List[Tuple[str, str]]],
    top_k: int,
    bm25_index: Optional[Tuple[BM25Okapi, List[Document]]],
    parent_store: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Run retrieval → rerank → dedup → context assembly.

    Returns (full_prompt, context_block, error_message, is_summarization).
    """
    t0 = time.monotonic()

    question_l      = question.lower()
    is_summarization = any(kw in question_l for kw in ["summarize", "summary", "overview", "give me a summary"])

    # Query rewriting (skipped for short queries — Improvement 4)
    detailed_q, expanded_q, keywords_q = _rewrite_query_with_llm(
        question, history, openai_model, openai_model_name, ollama_model, ollama_model_name
    )

    parts = []
    for v in [question, detailed_q, expanded_q, keywords_q]:
        v = (v or "").strip()
        if v and v not in parts:
            parts.append(v)
    retrieval_query = " \n".join(parts) if parts else question

    faiss_k = top_k * 12 if is_summarization else top_k * 6

    # For summarization, embed a broad coverage query rather than "summarize this
    # document" — the latter has no semantic match to any specific chunk, giving
    # artificially low scores that trigger the confidence gate incorrectly.
    if is_summarization:
        retrieval_query = "main topics key points overview introduction conclusion summary"

    try:
        faiss_results = vector_store.similarity_search_with_score(retrieval_query, k=faiss_k)
    except Exception as e:
        return None, None, f"Error retrieving relevant content: {e}", is_summarization

    # Hybrid retrieval with real BM25 scores (Improvement 5)
    if bm25_index is not None:
        try:
            bm25, bm25_docs = bm25_index
            bm25_results = _bm25_search(bm25, bm25_docs, retrieval_query, faiss_k)
            results = _reciprocal_rank_fusion([faiss_results, bm25_results])
        except Exception as exc:
            logger.warning("BM25 retrieval failed, falling back to FAISS only: %s", exc)
            results = faiss_results
    else:
        results = faiss_results

    if not results:
        return None, None, "No documents were retrieved for this question.", is_summarization

    kept_top_k = top_k * 2 if is_summarization else top_k
    kept = rerank_results(question, results, top_k=kept_top_k)

    if not kept:
        return None, None, "No relevant content found for this question.", is_summarization

    # Chunk deduplication (Improvement 2)
    kept = _deduplicate_chunks(kept)

    best_score = max(float(s) for _, s in kept)

    # Adaptive threshold (Improvement 3) — skipped for summarization because any
    # chunk from the document is valid context; the confidence gate would wrongly
    # reject results whose reranker score is low simply due to the generic query.
    global _session_best_scores
    _session_best_scores.append(best_score)
    _session_best_scores = _session_best_scores[-_MAX_SCORE_HISTORY:]
    threshold = _get_adaptive_threshold()
    conf_ok   = is_summarization or (best_score >= threshold)

    if not conf_ok:
        return None, None, (
            "The retrieved passages are not closely related enough to your question "
            "to answer confidently. Try rephrasing or uploading a more relevant document."
        ), is_summarization

    # Parent-window context (Improvement 1)
    context_block, sources_legend = _build_context_block(kept, is_summarization, parent_store)

    history      = history or []
    history_text = (
        "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:] if q and a)
        or "(none)"
    )

    system_prompt = _SUMMARIZATION_PROMPT if is_summarization else _get_qa_system_prompt()

    # Use proper chat message roles so Ollama/OpenAI apply the system prompt correctly.
    # Passing everything as a plain string wraps it all as a HumanMessage, which means
    # the model never activates its system-role behaviour (especially critical for Ollama).
    user_content = (
        f"--- CONVERSATION HISTORY ---\n{history_text}\n\n"
        f"--- CONTEXT ---\n{context_block}\n"
        f"{sources_legend}\n\n"
        f"--- QUESTION ---\n{question}"
    )
    full_prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    # Richer logging (Improvement 6)
    _log_rag_event(
        question   = question,
        top_scores = [float(s) for _, s in kept],
        kept_k     = len(kept),
        threshold  = threshold,
        confidence_ok = conf_ok,
        model_name = None,     # model not selected yet at this stage
        latency_s  = time.monotonic() - t0,
        answer_len = 0,
    )

    return full_prompt, context_block, None, is_summarization


# ── Public generation API ─────────────────────────────────────────────────────

def build_context_and_answer(
    vector_store: FAISS,
    question: str,
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
    history: Optional[List[Tuple[str, str]]] = None,
    polish: bool = False,
    top_k: int = 4,
    temperature: Optional[float] = None,
    bm25_index: Optional[Tuple[BM25Okapi, List[Document]]] = None,
    parent_store: Optional[Dict[str, str]] = None,
) -> str:
    if not question.strip():
        return "Please enter a question."

    t0 = time.monotonic()

    full_prompt, context_block, error, _ = _retrieve_and_build_context(
        vector_store, question,
        openai_model, openai_model_name,
        ollama_model, ollama_model_name,
        history, top_k, bm25_index, parent_store,
    )

    if error is not None:
        return error

    assert full_prompt is not None and context_block is not None

    selected_model, selected_name = _select_model(
        openai_model, openai_model_name, ollama_model, ollama_model_name, temperature
    )

    if selected_model is None:
        return (
            "No LLM model is available (neither OpenAI nor Ollama). "
            "Here is the retrieved context you can inspect manually:\n\n"
            + context_block
        )

    try:
        response = selected_model.invoke(full_prompt)
        answer   = response.content
        if not answer or len(answer.strip()) < 20:
            return (
                "The model returned an unhelpful answer. "
                "Here is the retrieved context instead:\n\n" + context_block
            )

        # Faithfulness check — append warning if answer seems poorly grounded
        if not polish:  # polish already does a second LLM pass; skip double-check
            answer = _append_faithfulness_warning(answer, context_block)

        if polish:
            refinement_messages = [
                SystemMessage(content=(
                    "You are an editor. Improve the following answer to make it clearer and more "
                    "readable. Do not introduce any new facts or speculation; only rephrase and "
                    "lightly restructure what is already there. Preserve all source citations "
                    "such as [1], [2], and any important numbers or names."
                )),
                HumanMessage(content=(
                    f"Question: {question}\n\n"
                    f"Original answer:\n{answer}\n\n"
                    "Return only the improved answer."
                )),
            ]
            try:
                refined = selected_model.invoke(refinement_messages)
                refined_answer = getattr(refined, "content", None)
                if refined_answer and len(refined_answer.strip()) >= 20:
                    answer = refined_answer
                else:
                    logger.warning("Polish step returned unusably short response; using original.")
            except Exception as exc:
                logger.warning("Polish step failed: %s", exc)

        _log_rag_event(
            question      = question,
            top_scores    = [],
            kept_k        = 0,
            threshold     = 0.0,
            confidence_ok = True,
            model_name    = selected_name,
            latency_s     = time.monotonic() - t0,
            answer_len    = len(answer),
        )
        return answer

    except Exception as e:
        return (
            f"Model error: {e}\n\n"
            "Here is the retrieved context, which you can inspect manually:\n\n"
            + context_block
        )


def stream_context_and_answer(
    vector_store: FAISS,
    question: str,
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
    history: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 4,
    temperature: Optional[float] = None,
    bm25_index: Optional[Tuple[BM25Okapi, List[Document]]] = None,
    parent_store: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """Stream the answer token-by-token.

    Yields string tokens from the LLM. On error or low confidence, yields a
    single error string and returns.
    """
    if not question.strip():
        yield "Please enter a question."
        return

    t0 = time.monotonic()

    full_prompt, context_block, error, _ = _retrieve_and_build_context(
        vector_store, question,
        openai_model, openai_model_name,
        ollama_model, ollama_model_name,
        history, top_k, bm25_index, parent_store,
    )

    if error is not None:
        yield error
        return

    assert full_prompt is not None and context_block is not None

    selected_model, selected_name = _select_model(
        openai_model, openai_model_name, ollama_model, ollama_model_name, temperature
    )

    if selected_model is None:
        yield (
            "No LLM model is available (neither OpenAI nor Ollama). "
            "Here is the retrieved context you can inspect manually:\n\n" + context_block
        )
        return

    answer_buf = ""
    try:
        for chunk in selected_model.stream(full_prompt):
            content = getattr(chunk, "content", "")
            if content:
                answer_buf += content
                yield content
    except Exception as e:
        yield f"\n\n[Streaming error: {e}]"
        return

    # Faithfulness check — yield warning token if answer seems poorly grounded
    score = _faithfulness_score(answer_buf, context_block)
    if score < -1.5:
        warning = (
            "\n\n---\n> ⚠ **Reliability notice:** This response may contain "
            "information not directly found in the uploaded document. "
            "Please verify key claims against the source."
        )
        yield warning
        answer_buf += warning

    _log_rag_event(
        question      = question,
        top_scores    = [],
        kept_k        = 0,
        threshold     = 0.0,
        confidence_ok = True,
        model_name    = selected_name,
        latency_s     = time.monotonic() - t0,
        answer_len    = len(answer_buf),
    )
