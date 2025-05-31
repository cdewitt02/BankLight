import requests
import argparse
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')    

# Classification examples for embedding fallback
CLASSIFICATION_EXAMPLES = [
    ("How much did I spend on groceries?", "sql"),
    ("What's my total spending last month?", "sql"),
    ("Show me my restaurant expenses", "sql"),
    ("What are my spending patterns?", "vector"),
    ("Find wasteful spending habits", "vector"),
    ("Show me unusual transactions", "vector"),
    ("Why are my food costs so high?", "hybrid"),
    ("Analyze my entertainment spending", "hybrid"),
    ("Help me budget better", "hybrid"),
    ("Hello", "none"),
    ("Thanks", "none")
]

class FinancialRAGAssistant:
    def __init__(self, model_name="qwen2.5:7b", db_config=None):
        self.model_name = model_name
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_config = db_config or {
            'host': DB_HOST,
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'port': int(DB_PORT) if DB_PORT else 5432
        }
        
        self.conn = None
        self.connect_to_db()
        
        self.example_embeddings = self.embed_model.encode([ex[0] for ex in CLASSIFICATION_EXAMPLES])
    
    def connect_to_db(self):
        """Establish PostgreSQL connection with pgvector support"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            print("✓ Connected to PostgreSQL database")
            
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✓ pgvector extension enabled")
                
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            self.conn = None
    
    def execute_sql(self, query: str, params=None):
        """Execute SQL query and return results"""
        if not self.conn:
            return "Database connection not available"
        
        print(f"[DEBUG] Executing SQL: {query} with params: {params}")

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return {'columns': columns, 'rows': rows}
                else:
                    return {'affected_rows': cur.rowcount}
        except psycopg2.Error as e:
            return f"SQL Error: {e}"
    
    def vector_search(self, query_embedding, limit=10):
        """Perform vector similarity search on transactions"""
        if not self.conn:
            return []
        
        embedding_list = query_embedding.tolist()
        
        query = """
        SELECT date, description, amount, account,
               embedding <-> %s as distance
        FROM transactions 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s
        LIMIT %s
        """
        
        result = self.execute_sql(query, [embedding_list, embedding_list, limit])
        if isinstance(result, dict) and 'rows' in result:
            return result['rows']
        return []
    
    def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Call Ollama API with given prompt"""
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    def pattern_classify(self, query: str) -> str:
        """Fast keyword-based classification"""
        query_lower = query.lower()
        
        sql_patterns = ['how much', 'total', 'sum', 'average', 'spent', 'last month', 'between', 'cost']
        vector_patterns = ['patterns', 'habits', 'wasteful', 'unusual', 'similar', 'behavior']
        
        sql_score = sum(1 for pattern in sql_patterns if pattern in query_lower)
        vector_score = sum(1 for pattern in vector_patterns if pattern in query_lower)
        
        if sql_score > vector_score and sql_score > 0:
            return 'sql'
        elif vector_score > sql_score and vector_score > 0:
            return 'vector'
        elif sql_score > 0 and vector_score > 0:
            return 'hybrid'
        else:
            return 'none'
    
    def embedding_classify(self, query: str) -> str:
        """Embedding-based classification fallback"""
        query_embedding = self.embed_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]
        best_match_idx = similarities.argmax()
        return CLASSIFICATION_EXAMPLES[best_match_idx][1]
    
    def classify_intent(self, query: str) -> str:
        """Hybrid classification: pattern matching + embedding fallback"""
        classification = self.pattern_classify(query)
        if classification == 'none':
            classification = self.embedding_classify(query)
        return classification
    
    def sql_rag(self, user_query: str) -> str:
        """Execute SQL-based RAG for analytical queries"""
        sql_prompt = f"""
        Generate a PostgreSQL query for this financial question: "{user_query}"
        
        The database contains a table named 'transactions' with the following schema:
        CREATE TABLE transactions (
            transaction_id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            account TEXT NOT NULL,
            embedding vector(384)
        );

        The query should focus on aggregating data such as totals, averages, or counts.

        The query cannot include insert, update, or delete operations. This is a read-only query.

        Return ONLY the raw SQL query as plain text. 
        Do NOT include markdown, backticks, comments, logging tags, or explanation.

        Example (correct):
        SELECT SUM(amount) AS total_spent FROM transactions;

        Example (incorrect):
        ```sql SELECT SUM(amount) AS total_spent FROM transactions; ```

        """
        
        sql_query = self.query_ollama(sql_prompt).strip()
        
        results = self.execute_sql(sql_query)
        
        response_prompt = f"""
        Question: {user_query}
        SQL Query: {sql_query}
        Results: {results}
        
        Provide a clear financial analysis based on this data.
        """
        
        return self.query_ollama(response_prompt)
    
    def vector_rag(self, user_query: str) -> str:
        """Execute vector search RAG for semantic queries"""
        # Create query embedding
        query_embedding = self.embed_model.encode([user_query])[0]
        
        # Search for similar transactions
        similar_transactions = self.vector_search(query_embedding, limit=5)
        
        response_prompt = f"""
        Question: {user_query}
        Similar transactions found: {similar_transactions}
        
        Analyze the patterns in these transactions and provide financial insights.
        """
        
        return self.query_ollama(response_prompt)
    
    def hybrid_rag(self, user_query: str) -> str:
        """Execute hybrid RAG combining SQL and vector search"""
        # Get structured data via SQL
        sql_prompt = f"""
        Generate a PostgreSQL aggregation query for: "{user_query}"
        Table: transactions (id, date, amount, merchant, category, description)
        Focus on totals, averages, or counts. Return only SQL.
        """
        sql_query = self.query_ollama(sql_prompt).strip()
        sql_results = self.execute_sql(sql_query)
        
        # Get contextual data via vector search
        query_embedding = self.embed_model.encode([user_query])[0]
        vector_results = self.vector_search(query_embedding, limit=5)
        
        response_prompt = f"""
        Question: {user_query}
        
        Quantitative analysis: {sql_results}
        Related transactions: {vector_results}
        
        Combine the numbers with transaction context to provide comprehensive financial advice.
        """
        
        return self.query_ollama(response_prompt)
    
    def direct_response(self, user_query: str) -> str:
        """Direct LLM response without RAG"""
        system_prompt = "You are a personal finance assistant. Provide general financial advice."
        return self.query_ollama(user_query, system_prompt)
    
    def process_query(self, user_query: str) -> str:
        """Main query processing with classification routing"""
        intent = self.classify_intent(user_query)
        print(f"[DEBUG] Classification: {intent}")
        
        if intent == 'sql':
            return self.sql_rag(user_query)
        elif intent == 'vector':
            return self.vector_rag(user_query)
        elif intent == 'hybrid':
            return self.hybrid_rag(user_query)
        else:  # none
            return self.direct_response(user_query)

def main():
    parser = argparse.ArgumentParser(description="Financial ChatBot with RAG and Query Classification")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model to use for Ollama")
    parser.add_argument("--host", default=DB_HOST, help="PostgreSQL host")
    parser.add_argument("--database", default=DB_NAME, help="Database name")
    parser.add_argument("--user", default=DB_USER, help="Database user")
    parser.add_argument("--password", default=DB_PASSWORD, help="Database password")
    parser.add_argument("--port", type=int, default=DB_PORT, help="Database port")
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.host,
        'database': args.database,
        'user': args.user,
        'password': args.password,
        'port': args.port
    }
    
    # Initialize assistant
    assistant = FinancialRAGAssistant(model_name=args.model, db_config=db_config)
    
    if not assistant.conn:
        print("Failed to connect to database. Exiting.")
        return
    
    print("Financial RAG Assistant initialized. Type 'exit' or 'quit' to end.")
    
    # Main interaction loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = assistant.process_query(user_input)
        print(f"Assistant: {response}")
    
    if assistant.conn:
        assistant.conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()