**Website Content Fetch vs FAISS Vector Search Flow**

Yes, that's exactly right. Here's the flow in `2_memory_based.ipynb`:

---

**Website content is fetched only once — at startup.**

```python
# Called once at the start of the program
vectorstore = process_website(WEBSITE_URL)
```

Inside `process_website()`:
1. `fetch_website_content(url)` — sends ONE HTTP request to https://www.snapy.ai/ and downloads the HTML
2. `scrape_website(url)` — parses HTML with BeautifulSoup to extract plain text
3. `text_splitter.split_text()` — splits the text into smaller chunks (1000 chars with 200 overlap)
4. `FAISS.from_texts(splits, embedding=embeddings)` — converts chunks to vector embeddings and stores them in an **in-memory FAISS index**

---

**For all subsequent queries, only the local FAISS index is searched.**

```python
retriever = vectorstore.as_retriever()  # points to the in-memory FAISS index

# Every user query → searches FAISS locally, NO website re-fetch
retrieved_docs = retriever.invoke(query)
```

No further HTTP requests, no re-downloading, no re-scraping happens.

---

**What lives where:**

| Component | When created | Scope | Purpose |
|---|---|---|---|
| Website content | Once (program start) | Temporary variable | Source text, discarded after chunking |
| FAISS index (in memory) | Once (program start) | RAM | Stores all chunk embeddings for similarity search |
| ConversationBufferMemory | Updated every query | RAM | Stores chat history for context |

---

**One caveat:** The FAISS index is NOT saved to disk in this code — it's purely in-memory. If you restart the kernel, the website will be fetched again. To make it truly persistent, you would add `vectorstore.save_local("faiss_index")` after creation and `FAISS.load_local(...)` on restart to skip the fetch.