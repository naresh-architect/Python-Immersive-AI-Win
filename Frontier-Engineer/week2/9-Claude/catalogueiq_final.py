#!/usr/bin/env python3
"""
CatalogueIQ — RAG-Powered Product Intelligence Assistant
ShopSmart India | Week 2 AI Engineering Capstone

Usage:
    python catalogueiq_final.py                          # interactive mode (default persona: shopper)
    python catalogueiq_final.py --persona shopper        # explicit shopper persona
    python catalogueiq_final.py --persona seller         # seller persona
    python catalogueiq_final.py --persona support        # support agent persona
    python catalogueiq_final.py --evaluate               # run RAGAS suite, save output/ragas_results.json
    python catalogueiq_final.py --rebuild                # re-ingest + re-embed even if index exists
    python catalogueiq_final.py --evaluate --rebuild     # rebuild then evaluate
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

MODEL           = "claude-sonnet-4-6"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DATA_DIR   = Path("data")
INDEX_DIR  = Path("data/index")
OUTPUT_DIR = Path("output")

# Retrieval settings
TOP_K_DEFAULT       = 4   # default chunks per query
TOP_K_COMPARATIVE   = 6   # higher k for comparative recommendation and multi-hop
MEMORY_WINDOW       = 5   # conversation turns to keep in memory

# ── DATA LOADING AND CHUNKING ─────────────────────────────────────────────────

def load_product_catalogue(csv_path: Path) -> list[Document]:
    """One Document per product row — preserves all specs in a single retrievable chunk."""
    import pandas as pd
    docs = []
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} products from {csv_path.name}")
    for _, row in df.iterrows():
        text = "\n".join([
            f"Product: {row.get('name', '')}",
            f"Brand: {row.get('brand', '')}",
            f"Category: {row.get('category', '')} > {row.get('subcategory', '')}",
            f"Price: ₹{row.get('price_inr', 'N/A')}",
            f"Rating: {row.get('rating', 'N/A')}/5 ({row.get('rating_count', 0)} reviews)",
            f"Specs: {row.get('specs', 'N/A')}",
            f"Warranty: {row.get('warranty_months', 0)} months",
            f"Returnable: {'Yes' if row.get('returnable', False) else 'No'} "
            f"({row.get('return_window_days', 0)}-day window)",
            f"Seller ID: {row.get('seller_id', '')} "
            f"(Verified: {'Yes' if row.get('seller_verified', False) else 'No'})",
            f"Description: {row.get('description', '')}",
        ])
        docs.append(Document(
            page_content=text,
            metadata={
                "source_file": csv_path.name,
                "file_type": "csv",
                "product_id": str(row.get("product_id", "UNKNOWN")),
                "product_name": str(row.get("name", "")),
                "category": str(row.get("category", "")),
                "price_inr": float(row.get("price_inr", 0)),
            }
        ))
    return docs


def load_markdown_doc(md_path: Path) -> list[Document]:
    """Split Markdown on headers first; then recursively on long sections."""
    text = md_path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "section_heading"), ("###", "subsection_heading")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = []
    for hdoc in header_splitter.split_text(text):
        if len(hdoc.page_content) > 1500:
            sub_docs = char_splitter.create_documents([hdoc.page_content], [hdoc.metadata])
            docs.extend(sub_docs)
        else:
            docs.append(hdoc)
    for doc in docs:
        doc.metadata.update({"source_file": md_path.name, "file_type": "markdown"})
    return docs


def load_buyer_faq(html_path: Path) -> list[Document]:
    """One Document per FAQ article, preserving section heading for metadata."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    docs = []
    current_section = "General"
    for el in soup.find_all(["h2", "article"]):
        if el.name == "h2":
            current_section = el.get_text(strip=True)
        elif el.name == "article":
            raw = " ".join(el.get_text(separator=" ", strip=True).split())
            docs.append(Document(
                page_content=raw,
                metadata={
                    "source_file": html_path.name,
                    "file_type": "html",
                    "question_number": el.get("id", "faq-unknown"),
                    "section_heading": current_section,
                }
            ))
    return docs


def load_review_summaries(csv_path: Path) -> list[Document]:
    """One Document per review-theme row."""
    import pandas as pd
    docs = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        text = (
            f"Product Review Summary — Product ID: {row.get('product_id', '')}\n"
            f"Theme: {row.get('theme', '')} | Sentiment: {row.get('sentiment', '')}\n"
            f"Mentions: {row.get('mention_count', 0)}\n"
            f"Summary: {row.get('summary', '')}\n"
            f"Representative quote: {row.get('representative_quote', '')}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source_file": csv_path.name,
                "file_type": "csv",
                "product_id": str(row.get("product_id", "")),
                "theme": str(row.get("theme", "")),
            }
        ))
    return docs


def ingest_all_documents() -> list[Document]:
    """Load and chunk all ShopSmart India knowledge base files."""
    all_docs: list[Document] = []

    file_loaders = [
        (DATA_DIR / "products.csv",         load_product_catalogue),
        (DATA_DIR / "category_taxonomy.md", load_markdown_doc),
        (DATA_DIR / "returns_policy.md",    load_markdown_doc),
        (DATA_DIR / "seller_onboarding.md", load_markdown_doc),
        (DATA_DIR / "buyer_faq.html",       load_buyer_faq),
        (DATA_DIR / "review_summaries.csv", load_review_summaries),
    ]

    for path, loader in file_loaders:
        if path.exists():
            try:
                docs = loader(path)
                all_docs.extend(docs)
                print(f"  ✓ {path.name}: {len(docs)} chunks")
            except Exception as e:
                print(f"  ✗ {path.name}: failed to load — {e}")
        else:
            print(f"  ⚠  {path.name}: not found (skipping)")

    print(f"\n  Total chunks ingested: {len(all_docs)}")
    return all_docs


# ── EMBEDDING AND INDEXING ────────────────────────────────────────────────────

def build_index(all_docs: list[Document]) -> tuple[LangchainFAISS, HuggingFaceEmbeddings]:
    """Embed all documents and build FAISS index. Saves to INDEX_DIR."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Indexing] Embedding {len(all_docs)} chunks with {EMBEDDING_MODEL}…")
    embedder_lc = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    t0 = time.time()
    vectorstore = LangchainFAISS.from_documents(all_docs, embedder_lc)
    elapsed = time.time() - t0
    print(f"[Indexing] Done in {elapsed:.1f}s")

    # Persist to disk
    vectorstore.save_local(str(INDEX_DIR))
    # Also save raw docs for metadata access
    with open(INDEX_DIR / "docs.pkl", "wb") as f:
        pickle.dump(all_docs, f)
    print(f"[Indexing] Index saved to {INDEX_DIR}/")
    return vectorstore, embedder_lc


def load_index() -> tuple[LangchainFAISS, HuggingFaceEmbeddings]:
    """Load cached FAISS index from INDEX_DIR."""
    embedder_lc = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = LangchainFAISS.load_local(
        str(INDEX_DIR), embedder_lc, allow_dangerous_deserialization=True
    )
    print(f"[Index] Loaded from {INDEX_DIR}/")
    return vectorstore, embedder_lc


def get_vectorstore(rebuild: bool = False) -> tuple[LangchainFAISS, HuggingFaceEmbeddings]:
    """Return vectorstore — rebuild from scratch or load from cache."""
    index_exists = (INDEX_DIR / "index.faiss").exists()

    if rebuild or not index_exists:
        if rebuild:
            print("[Rebuild] Re-ingesting all documents…")
        else:
            print("[Setup] No cached index found. Building from scratch…")
        docs = ingest_all_documents()
        return build_index(docs)
    else:
        return load_index()


# ── QUERY EXPANSION ───────────────────────────────────────────────────────────

SHOPSMRT_SYNONYMS = {
    "earphones":          ["earbuds", "TWS earbuds", "in-ear headphones"],
    "earbuds":            ["earphones", "TWS", "in-ear headphones"],
    "headphones":         ["over-ear headphones", "headset"],
    "mobile":             ["smartphone", "phone", "handset"],
    "kurtis":             ["kurta", "ethnic wear", "Indian wear"],
    "kurta":              ["kurti", "ethnic top"],
    "returns":            ["refund", "exchange", "send back"],
    "refund":             ["money back", "return"],
    "cheap":              ["affordable", "budget", "low price"],
    "noise cancellation": ["ANC", "active noise cancelling"],
    "ANC":                ["active noise cancellation", "noise cancelling"],
}


def static_expand(query: str) -> list[str]:
    """Apply static synonym expansion; return up to 3 variants."""
    variants = [query]
    ql = query.lower()
    for term, syns in SHOPSMRT_SYNONYMS.items():
        if term.lower() in ql:
            for syn in syns[:1]:
                expanded = re.sub(re.escape(term), syn, query, flags=re.IGNORECASE)
                if expanded not in variants:
                    variants.append(expanded)
    return variants[:3]


def llm_expand(query: str, llm: ChatAnthropic, n: int = 2) -> list[str]:
    """Use Claude to generate n semantically equivalent query variants.

    COST NOTE: one API call to Claude per user query.
    """
    prompt = (
        f"You are an Indian e-commerce search expert.\n"
        f"Rewrite this shopper query into {n} alternative phrasings preserving exact intent.\n"
        f"Consider: synonyms, Hinglish, abbreviations, Indian English.\n\n"
        f"Query: {query}\n\n"
        f"Respond ONLY with a JSON array of {n} strings. No markdown, no explanation."
    )
    # COST NOTE: API call to Claude
    try:
        response = llm.invoke(prompt)
        raw = re.sub(r"```json|```", "", response.content.strip()).strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [query] + [v for v in variants if v != query]
    except Exception:
        pass
    return static_expand(query)


def multi_retrieve(
    query: str,
    vectorstore: LangchainFAISS,
    llm: ChatAnthropic,
    top_k: int = TOP_K_DEFAULT,
    use_llm_expand: bool = True,
) -> list[Document]:
    """Expand query → retrieve per variant → deduplicate → return top_k."""
    # COST NOTE: llm_expand makes one API call if use_llm_expand=True
    variants = llm_expand(query, llm) if use_llm_expand else static_expand(query)

    seen: dict[str, tuple[Document, float]] = {}
    for variant in variants:
        results = vectorstore.similarity_search_with_score(variant, k=top_k)
        for doc, score in results:
            key = doc.metadata.get("product_id") or hash(doc.page_content[:80])
            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    ranked = sorted(seen.values(), key=lambda x: x[1])
    return [doc for doc, _ in ranked[:top_k]]


# ── RAG CHAIN SETUP ───────────────────────────────────────────────────────────

# Persona system prompts — each tailors tone and focus for the user type
PERSONA_PROMPTS = {
    "shopper": """\
You are CatalogueIQ, a friendly and helpful shopping assistant for ShopSmart India.
Help the shopper find the right product, understand specs, compare options, and know their rights.

RULES:
- Never invent product specs. Only state what is in the retrieved context.
- Never invent policy rules. Always cite the exact policy section.
- If a product is not in the catalogue, say: "This product is not available on ShopSmart India."
- Cite sources: [Source: <filename>, Product ID: <id> or Section: <heading>]
- For comparative queries, compare ALL products from the retrieved context — do not mention only one.
- Use ₹ for Indian Rupee amounts.
- Keep the tone warm, conversational, and helpful.

Context:
{context}""",

    "seller": """\
You are CatalogueIQ, a knowledgeable business assistant for ShopSmart India sellers.
Help the seller understand listing requirements, policies, pricing rules, and compliance.

RULES:
- Never invent policy rules. Cite the exact section from the seller onboarding guide.
- If a requirement is not in the retrieved context, say so and suggest consulting the Help Centre.
- Always cite: [Source: <filename>, Section: <heading>]
- Use ₹ for Indian Rupee amounts.
- Be precise and professional — sellers need accurate, actionable guidance.

Context:
{context}""",

    "support": """\
You are CatalogueIQ, an internal support agent assistant for ShopSmart India.
Help support agents resolve customer issues by retrieving accurate product and policy information.

RULES:
- Prioritise accuracy. Never guess. If context is missing, say: "I need to escalate — this information is not in the knowledge base."
- Always cite sources with full metadata: [Source: <filename>, Product ID: <id> or Section: <heading> or FAQ: <question_number>]
- For returns/warranty disputes, cite both the product's return_window_days and the A-to-Z Guarantee policy.
- Flag any ambiguity in the policy rather than guessing.
- Use ₹ for Indian Rupee amounts.

Context:
{context}""",
}

HUMAN_TEMPLATE = "{question}"


def build_conv_chain(
    vectorstore: LangchainFAISS,
    llm: ChatAnthropic,
    persona: str = "shopper",
    top_k: int = TOP_K_DEFAULT,
) -> tuple[ConversationalRetrievalChain, ConversationBufferWindowMemory]:
    """Build ConversationalRetrievalChain with persona system prompt and window memory."""
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    system_prompt_text = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["shopper"])
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
    )
    return chain, memory


def format_sources(source_docs: list[Document]) -> str:
    """Format retrieved source documents as a compact citation string."""
    refs = []
    for doc in source_docs:
        m = doc.metadata
        src = m.get("source_file", "?")
        if m.get("product_id"):
            refs.append(f"{src} [Product: {m['product_id']}]")
        elif m.get("section_heading"):
            refs.append(f"{src} [§ {m['section_heading']}]")
        elif m.get("question_number"):
            refs.append(f"{src} [{m['question_number']}]")
        else:
            refs.append(src)
    return " | ".join(refs) if refs else "No sources"


# ── CONVERSATION MEMORY ───────────────────────────────────────────────────────

def interactive_loop(
    vectorstore: LangchainFAISS,
    llm: ChatAnthropic,
    persona: str = "shopper",
) -> None:
    """Run the interactive multi-turn conversation loop."""
    chain, memory = build_conv_chain(vectorstore, llm, persona=persona)

    persona_labels = {
        "shopper": "🛒 ShopSmart India — Shopper Assistant",
        "seller":  "🏪 ShopSmart India — Seller Assistant",
        "support": "🎧 ShopSmart India — Support Agent",
    }
    print(f"\n{'='*60}")
    print(f"  {persona_labels.get(persona, 'CatalogueIQ')}")
    print(f"  Model: {MODEL} | Persona: {persona}")
    print(f"  Type 'quit' or 'exit' to end the session.")
    print(f"  Type 'reset' to clear conversation memory.")
    print(f"{'='*60}\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("[Session ended]")
            break
        if user_input.lower() == "reset":
            memory.clear()
            print("[Memory cleared — starting fresh conversation]\n")
            turn = 0
            continue

        turn += 1
        t0 = time.time()

        # COST NOTE: Two API calls per turn (condense question + generate answer)
        result = chain.invoke({"question": user_input})

        elapsed = time.time() - t0
        answer = result.get("answer", "[No answer generated]")
        sources = result.get("source_documents", [])

        print(f"\nCatalogueIQ: {answer}")
        print(f"[Sources: {format_sources(sources)}]  [{elapsed:.1f}s]\n")


# ── RAGAS EVALUATION ──────────────────────────────────────────────────────────

def run_ragas_evaluation(
    vectorstore: LangchainFAISS,
    llm: ChatAnthropic,
) -> dict:
    """
    Run full RAGAS evaluation against the golden test set.
    Saves results to output/ragas_results.json.

    COST NOTE: ~130 API calls to Claude (generation + RAGAS judge calls).
    Estimated cost: $0.20–$0.40 at current pricing.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Load golden test set
    golden_path = OUTPUT_DIR / "golden_test_set.json"
    if not golden_path.exists():
        print(f"[RAGAS] Golden test set not found at {golden_path}")
        print("[RAGAS] Run NB-02 first to generate the golden test set.")
        return {}

    with open(golden_path, encoding="utf-8") as f:
        golden_test_set = json.load(f)

    print(f"\n[RAGAS] Loaded {len(golden_test_set)} golden test questions")
    print("[RAGAS] Generating answers with baseline pipeline (top_k=3)…")

    # ── Baseline: top_k=3 ──────────────────────────────────────────────────
    baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    baseline_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PERSONA_PROMPTS["shopper"].replace("{context}", "{context}") +
                 "\n\nQuestion: {question}\n\nAnswer:",
    )
    baseline_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=baseline_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are CatalogueIQ for ShopSmart India.\n"
                "Answer using ONLY the context. Never invent specs or policies.\n"
                "Cite sources: [Source: <file>, Product ID: <id> or Section: <heading>]\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )},
    )

    questions, ground_truths, query_types = [], [], []
    baseline_answers, baseline_contexts = [], []

    for item in golden_test_set:
        q = item["question"]
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        query_types.append(item["query_type"])
        print(f"  [baseline] {item['query_type']}: {q[:55]}…")
        # COST NOTE: API call to Claude
        result = baseline_chain.invoke({"query": q})
        baseline_answers.append(result["result"])
        baseline_contexts.append([d.page_content for d in result.get("source_documents", [])])

    # ── Improved: top_k=6 for comparative + multi-hop ─────────────────────
    improved_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    improved_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=improved_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are CatalogueIQ for ShopSmart India.\n"
                "Answer using ONLY the context. Never invent specs or policies.\n"
                "Cite sources: [Source: <file>, Product ID: <id> or Section: <heading>]\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )},
    )

    improved_answers = list(baseline_answers)
    improved_contexts = list(baseline_contexts)
    target_types = {"comparative_recommendation", "multi_hop"}

    print("\n[RAGAS] Re-running with top_k=6 for comparative + multi-hop queries…")
    for i, item in enumerate(golden_test_set):
        if item["query_type"] in target_types:
            print(f"  [improved] {item['query_type']}: {item['question'][:55]}…")
            # COST NOTE: API call to Claude
            result = improved_chain.invoke({"query": item["question"]})
            improved_answers[i] = result["result"]
            improved_contexts[i] = [d.page_content for d in result.get("source_documents", [])]

    # ── RAGAS evaluation ──────────────────────────────────────────────────
    ragas_llm = LangchainLLMWrapper(llm)
    hf_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embedder)

    print("\n[RAGAS] Running RAGAS evaluation on baseline (this takes 3–5 minutes)…")
    # COST NOTE: ~120 API calls for RAGAS judge
    baseline_dataset = Dataset.from_dict({
        "question": questions, "answer": baseline_answers,
        "contexts": baseline_contexts, "ground_truth": ground_truths,
    })
    baseline_results = evaluate(
        dataset=baseline_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm, embeddings=ragas_embeddings,
    )

    print("[RAGAS] Running RAGAS evaluation on improved pipeline…")
    improved_dataset = Dataset.from_dict({
        "question": questions, "answer": improved_answers,
        "contexts": improved_contexts, "ground_truth": ground_truths,
    })
    improved_results = evaluate(
        dataset=improved_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm, embeddings=ragas_embeddings,
    )

    b_df = baseline_results.to_pandas()
    i_df = improved_results.to_pandas()
    b_df["query_type"] = query_types
    i_df["query_type"] = query_types

    # ── Print results table ───────────────────────────────────────────────
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"\n{'─'*70}")
    print(f"RAGAS RESULTS — CatalogueIQ | Model: {MODEL}")
    print(f"{'─'*70}")
    print(f"{'Query Type':<32} {'Faith':>6} {'AnsRel':>7} {'CtxPrc':>7} {'CtxRec':>7}  [Baseline]")
    print(f"{'':32} {'':>6} {'':>7} {'':>7} {'':>7}  → Delta after improvement")
    print("─" * 70)

    results_by_type = {}
    for qt in ["product_factual", "policy_eligibility", "comparative_recommendation",
               "seller_policy", "multi_hop"]:
        b_sub = b_df[b_df["query_type"] == qt]
        i_sub = i_df[i_df["query_type"] == qt]
        if len(b_sub) == 0:
            continue
        b_scores = {m: b_sub[m].mean() for m in metrics}
        i_scores = {m: i_sub[m].mean() for m in metrics}
        results_by_type[qt] = {"baseline": b_scores, "improved": i_scores}

        print(f"{qt:<32} "
              f"{b_scores['faithfulness']:>6.3f} {b_scores['answer_relevancy']:>7.3f} "
              f"{b_scores['context_precision']:>7.3f} {b_scores['context_recall']:>7.3f}")

        delta_line = "".join(
            f"  {'▲' if i_scores[m]-b_scores[m]>0.01 else ('▼' if i_scores[m]-b_scores[m]<-0.01 else '→')}{i_scores[m]-b_scores[m]:+.3f}"
            for m in metrics
        )
        print(f"{'  Δ improvement':<32}{delta_line}")

    print("─" * 70)
    b_overall = {m: b_df[m].mean() for m in metrics}
    i_overall = {m: i_df[m].mean() for m in metrics}
    print(f"{'OVERALL':<32} "
          f"{b_overall['faithfulness']:>6.3f} {b_overall['answer_relevancy']:>7.3f} "
          f"{b_overall['context_precision']:>7.3f} {b_overall['context_recall']:>7.3f}")

    # ── Save results JSON ─────────────────────────────────────────────────
    output = {
        "metadata": {
            "model": MODEL, "embedding_model": EMBEDDING_MODEL,
            "baseline_top_k": 3, "improved_top_k": 6,
            "total_questions": len(golden_test_set),
        },
        "baseline": {"overall": b_overall, "by_query_type": {
            qt: v["baseline"] for qt, v in results_by_type.items()
        }},
        "improved": {"overall": i_overall, "by_query_type": {
            qt: v["improved"] for qt, v in results_by_type.items()
        }},
        "per_question": [
            {
                "question": questions[idx],
                "query_type": query_types[idx],
                "baseline_answer": baseline_answers[idx],
                "improved_answer": improved_answers[idx],
                "baseline_scores": {m: float(b_df.iloc[idx][m]) for m in metrics},
                "improved_scores": {m: float(i_df.iloc[idx][m]) for m in metrics},
            }
            for idx in range(len(questions))
        ],
    }

    out_path = OUTPUT_DIR / "ragas_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[RAGAS] Full results saved to {out_path}")
    return output


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CatalogueIQ — ShopSmart India RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--persona", choices=["shopper", "seller", "support"], default="shopper",
        help="System prompt persona (default: shopper)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run RAGAS evaluation suite and save output/ragas_results.json",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-ingest and re-embed all documents even if index exists",
    )
    args = parser.parse_args()

    # ── Validate environment ──────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found. Add it to a .env file.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or build index ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CatalogueIQ — ShopSmart India Product Intelligence")
    print(f"  Model: {MODEL} | Persona: {args.persona}")
    print(f"{'='*60}")

    vectorstore, _ = get_vectorstore(rebuild=args.rebuild)

    # COST NOTE: LLM is initialised here; each generate() call costs tokens.
    llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=1024)

    # ── Run evaluation or interactive mode ────────────────────────────────
    if args.evaluate:
        print("\n[Mode] RAGAS evaluation")
        run_ragas_evaluation(vectorstore, llm)
    else:
        print(f"\n[Mode] Interactive conversation — persona: {args.persona}")
        interactive_loop(vectorstore, llm, persona=args.persona)


if __name__ == "__main__":
    main()
