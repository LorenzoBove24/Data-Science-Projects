# backend/rag_single_agent.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
from .rag_utils import (
    _get_vector_db_dirs,
    _describe_databases,
    _decide_which_dbs,
    _build_agent_config_log,
)


# =====================================================================
# Context builder + similarity filtering (with logging)
# =====================================================================
def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
    chunks = []
    total = 0
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "")
        db_prefix = f"[DB: {db_name}] " if db_name else ""
        header = f"[DOC {i+1} | {db_prefix}source: {src}]\n"
        text = d.page_content
        piece = header + text + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _similarity_rank_and_filter(
    question: str,
    docs: List[Document],
    embedding_model,
    top_k: int,
    min_sim: float = None,
) -> Tuple[List[Document], str, Optional[List[float]]]:
    """
    Rank docs by cosine similarity and filter below min_sim.
    If min_sim is None, use a dynamic threshold: mean - std of similarities.
    Returns (filtered_docs, log_string, similarities_kept).
    """
    log_lines: List[str] = []

    if not docs:
        log_lines.append("No documents returned from base retriever.")
        return [], "\n".join(log_lines), None

    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    doc_texts = [d.page_content for d in docs]
    doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype="float32")

    q_norm = np.linalg.norm(q_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    denom = np.maximum(q_norm * doc_norms, 1e-8)
    sims = (doc_vecs @ q_vec) / denom

    num_raw = len(docs)
    sims_min = float(np.min(sims))
    sims_max = float(np.max(sims))
    sims_mean = float(np.mean(sims))
    sims_std = float(np.std(sims))


    # Dynamic threshold if not provided
    if min_sim is None:
        min_sim = sims_mean - sims_std
        log_lines.append(f"[Dynamic threshold] min_sim = mean - std = {min_sim:.3f}")

    indices = [i for i, s in enumerate(sims) if s >= min_sim]
    num_after_threshold = len(indices)

    if not indices:
        log_lines.append(
            f"Similarity filtering: {num_raw} raw docs → 0 kept "
            f"(threshold={min_sim:.3f}, "
            f"sim range=[{sims_min:.3f}, {sims_max:.3f}], mean={sims_mean:.3f})."
        )
        return [], "\n".join(log_lines), None

    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    final_docs = [docs[i] for i in indices_sorted]

    sims_kept = sims[indices_sorted]

    sims_kept_min = float(np.min(sims_kept))
    sims_kept_max = float(np.max(sims_kept))
    sims_kept_mean = float(np.mean(sims_kept))

    log_lines.append(
        "Similarity filtering + reranking:\n"
        f"- Raw docs from retriever: {num_raw}\n"
        f"- Docs above threshold {min_sim:.3f}: {num_after_threshold}\n"
        f"- Final top_k={top_k} docs kept: {len(final_docs)}\n"
        f"- Similarity stats (all raw): min={sims_min:.3f}, max={sims_max:.3f}, "
        f"mean={sims_mean:.3f}\n"
        f"- Similarity stats (kept):   min={sims_kept_min:.3f}, max={sims_kept_max:.3f}, "
        f"mean={sims_kept_mean:.3f}"
    )

    return final_docs, "\n".join(log_lines), list(sims_kept)



def _retrieve_documents_from_db(
    question: str,
    config: RAGConfig, # Presumo sia una tua classe
    embedding_model,
    db_name: str,
    db_path: str,
    target_countries: Optional[List[str]] = None,
) -> Tuple[List[Document], str]:
    """
    Retrieve docs from a single FAISS DB at db_path, single-query only.
    Returns (docs_kept, log_string).
    """
    log_lines: List[str] = [f"[DB {db_name}] path={db_path}"]

    vector_store = load_vector_store(db_path, embedding_model)

    k_base = max(config.top_k * 3, config.top_k)
    
    search_kwargs = {"k": k_base}
    
    if target_countries:
        if len(target_countries) == 1:
            # Ricerca standard per singolo Paese (più veloce)
            search_kwargs["filter"] = {"country": target_countries[0]}
        else:
            # Ricerca comparativa per più Paesi usando una funzione lambda.
            # Ritorna True se il 'country' del documento è nella nostra lista.
            search_kwargs["filter"] = lambda meta: meta.get("country") in target_countries
            
        log_lines.append(f"[DB {db_name}] Applicato filtro metadati: Paesi = {target_countries}")
    else:
        log_lines.append(f"[DB {db_name}] Nessun filtro Paese applicato, ricerca globale.")

    # Passiamo il nostro dizionario search_kwargs aggiornato al retriever
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    log_lines.append(f"[DB {db_name}] Base retriever k={k_base} (top_k={config.top_k}).")

    log_lines.append(f"[DB {db_name}] Multi-query retrieval DISABLED.")
    raw_docs = base_retriever.invoke(question)

    log_lines.append(f"[DB {db_name}] Raw docs from retriever: {len(raw_docs)}")


    docs, sim_log, sims_kept = _similarity_rank_and_filter(
        question=question,
        docs=raw_docs,
        embedding_model=embedding_model,
        top_k=config.top_k,
        min_sim=None,  # Use dynamic threshold
    )
    log_lines.append(sim_log)

    # Attach similarity scores to document metadata for UI explainability
    if docs and sims_kept is not None:
        for d, sim in zip(docs, sims_kept):
            d.metadata = d.metadata or {}
            d.metadata["similarity_score"] = float(sim)

    if not docs:
        log_lines.append(f"[DB {db_name}] Result: no docs kept after filtering.")
    else:
        log_lines.append(f"[DB {db_name}] Result: {len(docs)} doc(s) kept for context.")

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["db_name"] = db_name

    return docs, "\n".join(log_lines)


# =====================================================================
# Agentic decision: do we need retrieval?
# =====================================================================
def _decide_need_retrieval(
    question: str,
    config: RAGConfig,
    llm_backend: LLMBackend,
) -> Tuple[bool, str]:
    system_prompt = (
                """You are an expert routing assistant for a specialized legal AI. 
        You have access to a specific external database that contains ONLY documents regarding:
        1. Divorce law and past cases.
        2. Inheritance law and past cases.
        Jurisdictions covered: Italy, Estonia, and Slovenia ONLY.

        Your ONLY task is to analyze the user's query and decide whether an external search in this specific database is required and appropriate.

        You must output STRICTLY the word "YES" or "NO". Do not include any other text, explanation, or punctuation.

        CRITERIA FOR "YES" (Query matches our database):
        - The query asks about divorce, separation, alimony, child custody, inheritance, wills, or successions specifically in Italy, Estonia, or Slovenia.
        - The query asks for comparisons between these three countries on those specific topics.

        CRITERIA FOR "NO" (Query does NOT match our database):
        - OUT OF JURISDICTION: The query is about divorce/inheritance, but in a country not listed above (e.g., France, USA, Germany).
        - OUT OF SCOPE: The query is about a different legal domain (e.g., corporate law, criminal law, traffic tickets, copyright), regardless of the country.
        - GENERAL KNOWLEDGE: The query asks for broad definitions (e.g., "what is a will?", "what is common law?") without tying it to specific procedures in the supported countries.
        - CHAT/IRRELEVANT: The query is a greeting or unrelated to law.

        FEW-SHOT EXAMPLES:

        User: "What are the legal requirements for a mutual consent divorce in Estonia?"
        Assistant: YES

        User: "How are assets divided during a divorce in France?"
        Assistant: NO

        User: "Can you explain the inheritance tax brackets in Slovenia?"
        Assistant: YES

        User: "What happens if someone dies without a will in Italy?"
        Assistant: YES

        User: "What is the penalty for corporate tax evasion in Italy?"
        Assistant: NO

        User: "What does it mean when a contract is considered 'void' in general terms?"
        Assistant: NO

        User: "Compare the divorce process between Italy and Estonia."
        Assistant: YES

        Now, analyze the following query and output only YES or NO."""
    )

    user_prompt = f"Question:\n{question}\n\nAnswer YES or NO only."

    resp = llm_backend.chat(system_prompt, user_prompt).strip().lower()

    if "yes" in resp and "no" not in resp:
        return True, f"Retrieval decision: model answered '{resp}' → USE retrieval."
    if "no" in resp and "yes" not in resp:
        return False, f"Retrieval decision: model answered '{resp}' → NO retrieval."

    return True, f"Retrieval decision: ambiguous answer '{resp}' → default to USE retrieval."


# =====================================================================
# Helper: summarized Observation text (using content + LLM)
# =====================================================================
def _build_observation_text(
    question: str,
    need_retrieval: bool,
    used_db_names: List[str],
    docs: List[Document],
    llm_backend: LLMBackend,
) -> str:
    if not need_retrieval:
        return "No external vector databases were used; the answer relies on internal knowledge."

    if need_retrieval and not docs:
        if used_db_names:
            db_list = ", ".join(sorted(set(used_db_names)))
            return (
                f"Retrieval was attempted on databases: {db_list}, "
                "but no sufficiently relevant documents were found."
            )
        return "Retrieval was attempted, but no sufficiently relevant documents were found."

    doc_blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "unknown_db")
        snippet = d.page_content[:400].replace("\n", " ").strip()
        doc_blocks.append(
            f"[DOC {i}] db={db_name} | source={src}\n"
            f"Snippet: {snippet}\n"
        )

    docs_text = "\n\n".join(doc_blocks)
    db_list = ", ".join(sorted(set(used_db_names))) if used_db_names else "unknown"

    system_prompt = (
        "You are summarizing how retrieved documents from one or more vector databases "
        "help answer a user's question.\n"
        "You MUST be concise and high-level. Do NOT reveal detailed chain-of-thought.\n"
        "Your job:\n"
        "- Mention briefly WHICH databases were used.\n"
        "- In 2–4 bullet points, explain at a high level how the retrieved content "
        "is useful or relevant for answering the question.\n"
        "- Keep the explanation short."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Databases used: {db_list}\n\n"
        f"Retrieved documents (summarized):\n{docs_text}\n\n"
        "Now produce a SHORT observation in this format:\n\n"
        "Databases used: <comma-separated list>\n"
        "- bullet point 1\n"
        "- bullet point 2\n"
        "- (optional) bullet point 3\n"
    )

    explanation = llm_backend.chat(system_prompt, user_prompt)
    return explanation


# =====================================================================
# SINGLE-AGENT CORE (ReAct-style)
# =====================================================================
from typing import Tuple, List, Optional, Dict
# Assicurati che questi import siano presenti nel tuo file, assieme a Document, RAGConfig, ecc.

def _single_agent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Original ReAct-style single-agent RAG pipeline (no multi-agent supervisor).
    Updated with Multi-Country extraction and dynamic FAISS filtering.
    """
    llm_backend = LLMBackend(config)
    db_map = _get_vector_db_dirs(config)  # {db_name -> path}

    # ---- Thought: need retrieval? ----
    need_retrieval, decision_log = _decide_need_retrieval(
        question, config, llm_backend
    )

    retrieved_docs: List[Document] = []
    used_db_names: List[str] = []
    context = ""
    db_selection_log = ""
    per_db_logs: Dict[str, str] = {}

    # ---- Action: if needed, pick DBs & retrieve ----
    db_descriptions: Dict[str, str] = {}
    if need_retrieval:
        embedding_model = get_embedding_model(config)
        db_descriptions = _describe_databases(db_map, embedding_model)

        used_db_names, db_selection_log = _decide_which_dbs(
            question=question,
            db_map=db_map,
            db_descriptions=db_descriptions,
            llm_backend=llm_backend,
        )

        if used_db_names:
            # --- MODIFICA MULTI-COUNTRY: Estrazione lista Paesi ---
            country_prompt = (
                "Analyze the following user question. Identify which of these three countries "
                "are explicitly mentioned: Italy, Estonia, Slovenia. "
                "Output STRICTLY a comma-separated list of the mentioned countries (e.g., 'Italy, Estonia' or 'Slovenia'). "
                "If none of these are mentioned, output STRICTLY 'NONE'."
                f"\n\nQuestion: {question}"
            )
            extracted_raw = llm_backend.chat(
                system_prompt="You are a precise data extraction assistant.", 
                user_prompt=country_prompt
            ).strip()
            
            # Creiamo una lista sicura dei Paesi trovati
            valid_countries = ["Italy", "Estonia", "Slovenia"]
            target_countries = [c for c in valid_countries if c.lower() in extracted_raw.lower()]
            
            db_selection_log += f"\n[Country Extraction] Extracted filters: {target_countries or 'NONE'}"
            # ----------------------------------------------------------

            all_docs: List[Document] = []
            for db_name in used_db_names:
                db_path = db_map[db_name]
                docs_db, log_db = _retrieve_documents_from_db(
                    question=question,
                    config=config,
                    embedding_model=embedding_model,
                    db_name=db_name,
                    db_path=db_path,
                    target_countries=target_countries,  # <-- Passiamo la LISTA dei paesi
                )
                per_db_logs[db_name] = log_db
                all_docs.extend(docs_db)

            retrieved_docs = all_docs
            context = _build_context(retrieved_docs)
        else:
            need_retrieval = False  # model explicitly selected NONE

    # ---- Answer: main LLM call ----
    
    # --- Istruzione di sicurezza e comparazione ---
    safety_instruction = (
        "CRITICAL: If the user asks about specific countries, base your answer "
        "ONLY on documents referencing those countries. If the user asks about multiple "
        "countries, structure your answer to clearly compare them using ONLY the provided context. "
        "Ignore context from jurisdictions not asked about."
    )

    if config.agentic_mode == "react":
        system_prompt = (
            "You are an agentic reasoning legal assistant. "
            "You MUST answer the user's question ONLY using the provided context from retrieved documents. "
            "If the answer is not in the provided context, say so explicitly — do NOT add information "
            "from your general knowledge or invent an answer. "
            "In all cases, do not reveal your internal chain-of-thought; provide only a clear final answer. " 
            + safety_instruction
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append(
            "Provide a clear, concise final answer without exposing your internal steps."
        )
        user_prompt = "\n\n".join(user_parts)
    else:
        system_prompt = (
            "You are a strict and precise legal assistant. "
            "You MUST answer the user's question ONLY using the provided context from retrieved documents. "
            "If the answer is not in the provided context, say so explicitly — do NOT add information "
            "from your general knowledge or invent an answer. " 
            + safety_instruction
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append("Provide a concise, accurate answer.")
        user_prompt = "\n\n".join(user_parts)

    answer = llm_backend.chat(system_prompt, user_prompt)

    # ---- Optional ReAct-style trace + retrieval + agent config logs ----
    reasoning_trace: Optional[str] = None
    if config.agentic_mode == "react" and show_reasoning:
        if need_retrieval:
            thought_str = (
                "The agent analyzed the question to understand its topic and "
                "determined that consulting one or more vector databases would "
                "improve the answer."
            )
        else:
            thought_str = (
                "The agent analyzed the question and decided it could be answered "
                "reliably without consulting any external databases."
            )

        if need_retrieval and used_db_names:
            action_str = (
                "The agent chose to retrieve from the following databases: "
                + ", ".join(f"`{n}`" for n in used_db_names)
                + "."
            )
        elif need_retrieval and not used_db_names:
            action_str = (
                "The agent considered retrieval, but did not select any specific "
                "database for this question."
            )
        else:
            action_str = (
                "The agent skipped retrieval and relied solely on its own knowledge."
            )

        observation_str = _build_observation_text(
            question=question,
            need_retrieval=need_retrieval,
            used_db_names=used_db_names,
            docs=retrieved_docs,
            llm_backend=llm_backend,
        )

        per_db_log_block = ""
        if per_db_logs:
            for db_name, log in per_db_logs.items():
                per_db_log_block += f"\n\n[DB {db_name}]\n{log}"

        retrieval_log_block = (
            f"{decision_log}\n\n"
            f"{db_selection_log}\n"
            f"{per_db_log_block.strip()}"
        ).strip()

        agent_config_log = _build_agent_config_log(
            config=config,
            db_map=db_map,
            db_descriptions=db_descriptions if db_descriptions else None,
        )

        reasoning_trace = (
            f"**Thought**: {thought_str}\n\n"
            f"**Action**: {action_str}\n\n"
            f"**Observation**:\n{observation_str}\n\n"
            f"**Retrieval / Post-Retrieval Optimization Log**:\n"
            f"```text\n{retrieval_log_block}\n```\n\n"
            f"**Agent / DB Configuration**:\n"
            f"```text\n{agent_config_log}\n```"
        )

    return answer, retrieved_docs, reasoning_trace


# Public alias
def single_agent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return _single_agent_answer_question_core(question, config, show_reasoning)
