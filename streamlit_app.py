import streamlit as st
from engine import Engine

# Initialize the search engine
@st.cache_resource
def init_engine():
    return Engine("repo_data_files")

engine = init_engine()

# App title and description
st.title("File Search System")
st.write("Search through repository files and find relevant content based on your query.")

# Search interface
query = st.text_input("Enter your search query")
threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.5, 0.1)
bm25_weight = st.slider("BM25 weight", 0.0, 1.0, 0.6, 0.1)
embedding_weight = st.slider("Embedding weight", 0.0, 1.0, 0.4, 0.1)

if query:
    # Perform search
    results = engine.hybrid_search(
        query,
        bm25_weight=bm25_weight,
        embedding_weight=embedding_weight,
        threshold=threshold
    )
    use_llm = st.checkbox("Use AI to refine results", value=False)
    if use_llm:
        results = engine.filter_using_llm(query, results)
    
    if not results:
        st.warning("No results found above the threshold.")
    else:
        st.success(f"Found {len(results)} relevant results")
        
        # Display results
        def get_file_emoji(path):
            ext = path.lower().split('.')[-1]
            emoji_map = {
                'py': '🐍',
                'txt': '📝',
                'md': '📑',
                'log': '📋',
                'cfg': '⚙️',
                'ipynb': '📓',
                'json': '📊',
                'yaml': '⚙️',
                'yml': '⚙️', 
                'csv': '📊',
                'jsonl': '📊'
            }
            return emoji_map.get(ext, '📄')

        for idx, result in enumerate(results):
            file_emoji = get_file_emoji(result['path'])
            with st.expander(f"{idx + 1}:  {result['path']} {file_emoji}"):
                st.text("Preview:")
                st.code(result['snippet'])
else:
    st.info("Enter a query to search through the files.")
