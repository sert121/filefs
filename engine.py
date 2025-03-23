import os
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')  # Add this line before using word_tokenize
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()



client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

class Engine:
    def __init__(self, root_folder, allowed_extensions=None):
        self.root_folder = "repo_data_files"
        self.allowed_extensions = allowed_extensions or ['.txt', '.md', '.log', '.cfg', '.py', '.ipynb', '.json', '.yaml', '.yml', '.csv', '.jsonl', '.jsonl.gz', '.jsonl.bz2', '.jsonl.zip', '.jsonl.tar', '.jsonl.tar.gz', '.jsonl.tar.bz2', '.jsonl.tar.zip']
        self.documents = []
        self.file_paths = []
        self.tokenized_docs = []

        # initializing the models to be used
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # data matrices
        self.tfidf_matrix = None
        self.embeddings = None

        # load and process files
        self.load_documents()
        self.build_indexes()

    def get_all_files(self):
        """recursively collects all files with specified extensions."""
        files = []
        for dirpath, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in self.allowed_extensions):
                    files.append(os.path.join(dirpath, filename))
        return files

    def read_file(self, filepath):
        """safely read a file's content."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Skipping file {filepath}: {e}")
            return ""

    def load_documents(self):
        """loads file contents from directory into memory."""
        files = self.get_all_files()
        print(f"Found {len(files)} files.")
        for file in files:
            content = self.read_file(file)
            if content.strip():  # skip empty files
                self.file_paths.append(file)
                self.documents.append(content)
        print(f"Loaded {len(self.documents)} readable documents.")

    def build_indexes(self):
        """build bm25, tf-idf, and embedding indexes."""
        print("Building indexes...")
        # ombine filename with content for indexing
        self.tokenized_docs = [
            word_tokenize(doc.lower() + " " + path.lower()) 
            for doc, path in zip(self.documents, self.file_paths)
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # tf-idf
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)

        # embeddings
        self.embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=True)

        print("Indexes built successfully.")

    @staticmethod
    def normalize(scores):
        """normalize a score array between 0 and 1."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def bm25_search(self, query):
        """return normalized bm25 scores for a query."""
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        return self.normalize(np.array(bm25_scores))

    def tfidf_search(self, query):
        """return cosine similarity scores for a query."""
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return cosine_sim  # already between 0 and 1

    def embedding_search(self, query):
        """return normalized cosine similarity scores for a query embedding."""
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(query_embedding, self.embeddings).squeeze().cpu().numpy()
        return self.normalize(cosine_sim)

    def hybrid_search(self, query, bm25_weight=0.6, embedding_weight=0.4, threshold=0.5):
        """performs a hybrid search combining bm25 and embedding similarity."""
        bm25_scores = self.bm25_search(query)
        embedding_scores = self.embedding_search(query)

        # combine scores
        final_scores = bm25_weight * bm25_scores + embedding_weight * embedding_scores
        results = []

        # collect results above threshold
        for idx, score in enumerate(final_scores):
            if score >= threshold:
                results.append({
                    "id": idx,
                    'path': self.file_paths[idx],
                    'score': score,
                    'snippet': self.documents[idx][:300]  # first 300 characters
                })

        # sort descending by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def filter_using_llm(self, query, results):
        """search for the query and display results."""
        # Create mapping from id to result for efficient lookups
        id_to_result = {result['id']: result for result in results}

        if not results:
            print("No results found above the threshold.")
        else:
            for res in results[:]:
                context_for_llm = []
                for res in results[:]:
                    context_for_llm.append({
                        'file': res['path'],
                        'id': res['id'],
                        'snippet': res['snippet']
                    })
            search_results_str = "\n\n".join([
                f"File Name: {r['file']}"
                f"\nSnippet: {r['snippet']}"
                f"\nid: {r['id']}"
                for r in context_for_llm
            ])

            prompt = f"""
            You are a helpful assistant that recommends files based on a user's query. You can understand the user query, as well as the corresponding file content.

            You will be provided with the existing search results and the user query. Your task is to provide your top 10 recommendations with brief explanations, ordered from highest to lowest relevance. Output in JSON format.

            ## Context
            Search Results:

            {search_results_str}

            User Query:
            {query}

            ## Example Output Format:
            {{
                "recommendations": [
                    {{
                        "file": "<insert_filename>",
                        "explanation": "<insert_explanation>",
                        "id": "<retain_id_from_context>"
                    }}
                    ...
                ]
            }}
            """

            response = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            recommendations = response.choices[0].message.content
            print(recommendations)

            import json
            recommendations_dict = json.loads(recommendations)
            
            mapped_recommendations = []
            for rec in recommendations_dict.get('recommendations', []):
                mapped_recommendations.append({
                    'id': rec['id'],
                    'path': rec['file'],
                    'score': id_to_result[rec['id']]['score'], 
                    'snippet': id_to_result[rec['id']]['snippet']
                })

            return mapped_recommendations


