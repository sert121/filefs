# FileFS

```
.
├── engine.py               # Core engine or processing script for indexing the files
├── README_workflow.md      # This File
├── requirements.txt        # Python package dependencies
└── streamlit_app.py        # Streamlit app for frontend or UI
└── repo_data_files         # mock generated files for indexing
```

---

# setup instructions

### 1. install `uv` 
```bash
pip install uv
```

### 2. install project dependencies with `uv`
```bash
uv pip install -r requirements.txt
```

Or, if you're using the `pyproject.toml`:
```bash
uv pip install -r requirements.txt -p pyproject.toml
```
Add openrouter key to a .env
OR `export OPENROUTER_API_KEY='openrouterkey'`

# usage
```bash
streamlit run streamlit_app.py # this indexes and then launches the app
```
here's a preview of the app  
![image](https://github.com/user-attachments/assets/9a908b55-d5da-4721-a590-3fd3303092cc)



## approach to calculating relevance

the search engine implements a **hybrid search** using a combination of three different ranking techniques:

1. **BM25**  
   - uses a **tokenized representation** of documents and a query.  
   - computes the **BM25 score**, a ranking function designed to score documents based on term frequency and inverse document frequency.  
   - results are normalized between **0 and 1** to allow combination with other methods.

2. **TF-IDF + Cosine Similarity**  
   - uses **TF-IDF vectorization** to represent documents and queries numerically.  
   - computes **cosine similarity** between the query and each document in the repository.  
   - higher similarity scores indicate stronger relevance.

3. **Embedding Search**  
   - uses **pre-trained embedding models (`all-MiniLM-L6-v2`)** to capture **semantic meaning** beyond simple word matching.  
   - computes **cosine similarity** between the query's embedding and document embeddings.  
   - resuots are normalized between **0 and 1** 

### **working of hybrid search**
- A weighted combination of **BM25 and embeddings-based similarity** is used to rank documents.
- The final relevance score is computed as:
$$
\text{Final Score} = \alpha \cdot \text{BM25 Score} + \beta \cdot \text{Embedding Score} \quad ( \text{this can range from } 0 \text{ to } 2 )
$$
  where **α = 0.6** and **β = 0.4** (default values, tunable),.
- docs scoring **above a threshold (0.5)** are considered relevant, but the relevance threshold can also be modified .

### **filtering using an llm (optional)**
- a LLM is used as a **final refinement step**.  
- it receives the **top-ranked search results** along with the user query and **re-ranks** them based on contextual understanding.  

---

## potential improvements

### 1. adaptive weight tuning
- the weights (α for bm25, β for embeddings) are currently fixed.  
- implementing adaptive tuning based on query type (e.g., short queries rely more on bm25, long queries on embeddings).

### 2. multi-stage re-ranking
- introduce a second-pass re-ranking model, such as a fine-tuned llm, to refine the search results.  
- could be based on learned ranking models like lambdamart or colbert.

### 3. streaming & caching optimization
- improve efficiency by:
  - incremental indexing instead of rebuilding indexes on every run.
  - disk-based bm25 caching for large datasets.

### 4. fuzzy finding and transforming search queries
- if we feel that that the search query is not good enough/contextual by the user, we can transform using a small LM or use fzf to generate possible searches. 

### 5. scaling
- although the current approach scales with larger number of documents (we only index once), the embedding is quite small, and the bm25 algorithm is quite performant, the only cost is to ingest the final rankings with an llm (which is minimal as we only rerank based on the llm). but it can be made more performant by caching 'good queries' or frequent queries across the system.
