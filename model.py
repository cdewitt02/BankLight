import requests
import argparse
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import readline

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

VALID_CLASSIFICATIONS = ['sql', 'vector', 'hybrid', 'none']

class FinancialRAGAssistant:
    def __init__(self, model_name="llama3.1:8b", db_config=None):
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
        self.conversation_history = []
    
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
    
    def llm_classify_intent(self, user_query: str, conversation_history: list = None) -> str:
        """Classify user intent using an LLM, considering the conversation history for context."""
        
        formatted_history_for_prompt = ""
        if conversation_history:
            formatted_history_for_prompt += "\n--- Recent Conversation History (for context) ---\n"
            # Format last few turns (e.g., last 3 turns = 6 messages, or fewer if history is short)
            # We take up to `max_history_messages_for_prompt` (e.g. 9 from process_query, so ~4-5 turns)
            for turn in conversation_history: # conversation_history is already sliced in process_query
                formatted_history_for_prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
            formatted_history_for_prompt += "--- End of Recent Conversation History ---\n\n"

        # The specific instruction about affirmative follow-ups needs to refer to the *last assistant message* from history
        last_asst_msg_guidance = ""
        if conversation_history and conversation_history[-1]["role"] == "assistant":
            last_asst_msg = conversation_history[-1]["content"]
            last_asst_msg_guidance = f"""
If the User Query is a simple affirmative (e.g., "yes", "yep", "sure", "ok", "alright") OR a short phrase that directly relates to a suggestion/question in the *immediately preceding* Assistant message below, AND that Assistant message ended by offering to provide more details, show a list, or perform a related action (which would typically be a 'sql' or 'vector' type action):
Then, your primary goal is to determine the intent as if the user had explicitly asked for that action. For example:
- If Assistant last asked: "Would you like to see the transactions?" (and this is part of the history) and User now says: "yes", the intent is likely 'sql' (to list transactions).
Treat these short follow-ups as a confirmation of the assistant's most recent offer.
Immediately Preceding Assistant Message (for context of this rule):
'''{last_asst_msg}'''
"""

        prompt = f"""{formatted_history_for_prompt}{last_asst_msg_guidance}
You are an expert query classifier for a financial assistant.
Based on the User Query (and the Recent Conversation History, if provided), determine the best approach to answer it.

User Query:
"{user_query}"

The database contains a table named 'transactions' with the following relevant schema for user queries:
- date (DATE): The date of the transaction.
- description (TEXT): A description of the transaction (e.g., "A purchase of items from Wal-Mart in Roanoke, Texas", "Salary deposit"). This field might be used for categorical queries if the `category` field is not specific enough or for keyword searches on descriptions.
- amount (REAL): The monetary value of the transaction. Positive for income/deposits, negative for expenses/withdrawals.
- account (TEXT): The account associated with the transaction (e.g., "Checking", "Savings").
- category (TEXT): The assigned category of the transaction (e.g., "GROCERIES", "RESTAURANT", "HOUSING/UTILITIES", "MISCELLANEOUS"). This is the primary field for categorical queries.

Available approaches:
- 'sql': For questions requiring precise data lookups, calculations (sum, average, count), or filtering of transactions based on specific criteria (dates, amounts, descriptions, accounts, categories). Note: For the 'amount' field, positive values represent deposits/income, and negative values represent withdrawals/spending. Examples: "How much did I spend on groceries last month?" (implies negative amounts and category 'GROCERIES'), "What was my total income in 2023?" (implies positive amounts), "Show all withdrawals over $50 in the RESTAURANT category."
- 'vector': For questions seeking patterns, trends, anomalies, or insights based on semantic similarity of transactions, especially when categories are too broad or the query is about the nature of the spending described in the `description`. Examples: "What are my unusual spending habits?", "Show me transactions similar to my trip to Italy.", "Identify potential wasteful spending.", "What are my general spending patterns for online subscriptions?"
- 'hybrid': For complex questions that require both specific data lookups (like SQL based on category, date, amount) AND broader pattern analysis or contextual understanding (like vector search on descriptions). Examples: "Analyze my spending in the ENTERTAINMENT category last quarter and suggest areas to save by showing related transactions.", "Compare my current travel expenses (by category and amount) to similar past trips and highlight differences.".
- 'none': For greetings, conversational fillers, or questions clearly not related to financial data analysis or the user's transactions. Examples: "Hello", "Thanks", "How are you?", "What is the capital of France?".

Your response MUST be one of the following exact strings: sql, vector, hybrid, none.
Do not add any explanation or other text.
"""
        
        llm_response_raw = self.query_ollama(prompt)

        if llm_response_raw.startswith("Error:"):
            print(f"[WARN] LLM classification call failed: {llm_response_raw}")
            return 'none'

        classification = llm_response_raw.strip().lower()

        if classification not in VALID_CLASSIFICATIONS:
            print(f"[WARN] LLM returned unexpected classification: '{classification}'. Defaulting to 'none'.")
            # You could add a fallback to self.embedding_classify(user_query) here if desired
            return 'none'
        
        print(f"[DEBUG] LLM Classification: {classification}")
        return classification

    def sql_rag(self, user_query: str, conversation_history: list = None) -> str:
        """Execute SQL-based RAG for analytical queries, using conversation history for context."""
        
        query_to_base_sql_on = user_query
        additional_context_info = ""
        formatted_history_for_sql_prompt = ""

        is_affirmative_follow_up = (
            conversation_history and 
            conversation_history[-1]["role"] == "assistant" and
            user_query.lower() in ["yes", "yep", "sure", "ok", "okay", "alright", "please do", "do it", "show me", "tell me", "list them"]
        )

        if is_affirmative_follow_up:
            query_to_base_sql_on = conversation_history[-1]["content"]
            additional_context_info = (
                f"CONTEXT: The user's literal input was \"{user_query}\", which is an affirmative response. "
                f"This means the user wants to proceed with the action or answer related to the assistant's previous statement (provided below as the main query). "
                f"Therefore, the PostgreSQL query you generate MUST address that previous statement."
            )
        else:
            additional_context_info = "Generate a PostgreSQL query for the following financial question:"
        
        if conversation_history:
            formatted_history_for_sql_prompt += "\n--- Recent Conversation History (for SQL context) ---\n"
            for turn in conversation_history:
                formatted_history_for_sql_prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
            formatted_history_for_sql_prompt += "--- End of Recent Conversation History ---\n\n"

        sql_prompt = f"""
        {formatted_history_for_sql_prompt}
        {additional_context_info}
        '''{query_to_base_sql_on}'''

        --- Database Schema and SQL Generation Rules ---
        The database contains a table named 'transactions' with the following schema:
        CREATE TABLE transactions (
            transaction_id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            account TEXT NOT NULL,
            category VARCHAR(50),
            embedding vector(384)
        );

        When referring to spending, expenses, or costs, this means transactions with a negative 'amount'.
        When referring to income, deposits, or earnings, this means transactions with a positive 'amount'.
        If the user asks about spending, ensure the query sums negative amounts (or sums the absolute values of negative amounts if total spending is requested as a positive number).
        If the user asks about income, ensure the query sums positive amounts.
        If the user mentions a specific category (e.g., "groceries", "entertainment"), use the `category` column for filtering (e.g., `WHERE category = 'GROCERIES'`). The available categories are typically: HOUSING/UTILITIES, GROCERIES, GAS, RESTAURANT, ENTERTAINMENT, MERCHANDISE & SUPPLIES, MISCELLANEOUS.

        The query should focus on aggregating data such as totals, averages, or counts.
        If the user asks for a ranking by time period (e.g., "rank months by spending"), ensure the query extracts the time period correctly (e.g., using TO_CHAR(date, 'YYYY-MM') for month).

        The query cannot include insert, update, or delete operations. This is a read-only query.

        Return ONLY the raw SQL query as plain text. 
        Do NOT include markdown, backticks, comments, logging tags, or explanation.

        Example (correct for simple sum):
        SELECT SUM(amount) AS total_spent FROM transactions;

        Example (correct for monthly ranking):
        SELECT TO_CHAR(date, 'YYYY-MM') AS month, SUM(amount) AS total_spent 
        FROM transactions 
        GROUP BY TO_CHAR(date, 'YYYY-MM') 
        ORDER BY total_spent DESC;

        Example (incorrect):
        ```sql SELECT SUM(amount) AS total_spent FROM transactions; ```

        """
        
        sql_query = self.query_ollama(sql_prompt).strip()
        
        results = self.execute_sql(sql_query)
        
        response_prompt = f"""
        User Question: {user_query}
        SQL Query Executed: {sql_query}
        SQL Query Results: {results}

        You are a helpful financial assistant. Based on the user's question and the SQL query results:
        Remember: In the 'transactions' table, positive amounts are income/deposits, and negative amounts are expenses/withdrawals.
        1.  Provide a direct and conversational answer to the user's question. Use appropriate financial terms (e.g., "spent", "earned", "deposited", "withdrew") based on the nature of the amounts.
        2.  If the SQL results are a single value (e.g., a count, sum, average), keep your answer concise.
            Example for a single value: "You have made a total of 43 transactions."
        3.  If the SQL results contain multiple rows that represent a list, ranking, or grouped data (e.g., spending by category, transactions per month), try to format the core data in a simple, human-readable, text-based table if it makes sense for clarity. Summarize key insights if the table is very long.
            Example for complex data (not a table): "Looking at your transactions, you were most active on May 27th with 5 transactions. Several other days in April and May also saw 2-3 transactions each."
        4.  Conclude by suggesting one relevant follow-up question the user might be interested in.
        Keep the tone friendly and helpful.

        Let's break down how to formulate the answer:
        - Start by directly addressing the user's question with the main finding.
        - If the data is suitable for a table (like a list or ranking), present the key data in a simple text table.
        - If the data is complex but not easily table-formatted, briefly elaborate on the main patterns or provide highlights.
        - Finish with a natural-sounding follow-up question.

        Example (Concise for single value):
        User Question: How many transactions did I make last week?
        SQL Query Executed: SELECT COUNT(*) FROM transactions WHERE date >= '2023-03-01' AND date <= '2023-03-07';
        SQL Query Results: {{"columns": ["count"], "rows": [[5]]}}

        Answer:
        You made 5 transactions last week.
        Would you like to see the details of these transactions?

        Example (Summarized for complex data/ranking, with table-like format for lists):
        User Question: Rank the months by how much I spent.
        SQL Query Executed: SELECT TO_CHAR(date, 'YYYY-MM') AS month, SUM(amount) AS total_spent FROM transactions GROUP BY TO_CHAR(date, 'YYYY-MM') ORDER BY total_spent DESC;
        SQL Query Results: {{"columns": ["month", "total_spent"], "rows": [["2023-04", -5500.00], ["2023-03", -4200.50], ["2023-05", -3100.75]]}}

        Answer:
        Here's a ranking of your spending by month:
        Month    | Total Spent
        ---------|--------------
        2023-04  | $5500.00 
        2023-03  | $4200.50 
        2023-05  | $3100.75 

        Would you like a more detailed breakdown for any of these months?

        Example (Summarized for general complex data including category):
        User Question: What were my top spending categories last month?
        SQL Query Executed: SELECT category, SUM(amount) as total_spent FROM transactions WHERE date >= '2023-03-01' AND date <= '2023-03-31' AND amount < 0 GROUP BY category ORDER BY total_spent ASC LIMIT 3;
        SQL Query Results: {{"columns": ["category", "total_spent"], "rows": [["GROCERIES", -150.75], ["DINING", -120.50], ["TRANSPORT", -80.00]]}}

        Answer:
        Last month, your top spending categories were Groceries at $150.75, Dining at $120.50, and Transport at $80.00.
        Would you like to explore the transactions for any of these categories in more detail?
        """
        
        return self.query_ollama(response_prompt)
    
    def vector_rag(self, user_query: str, conversation_history: list = None) -> str:
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
    
    def hybrid_rag(self, user_query: str, conversation_history: list = None) -> str:
        """Execute hybrid RAG combining SQL and vector search, using conversation history."""
        
        query_to_base_sql_on = user_query
        additional_context_info_for_sql = ""
        formatted_history_for_sql_prompt = ""

        is_affirmative_follow_up = (
            conversation_history and 
            conversation_history[-1]["role"] == "assistant" and
            user_query.lower() in ["yes", "yep", "sure", "ok", "okay", "alright", "please do", "do it", "show me", "tell me", "list them"]
        )

        if is_affirmative_follow_up:
            query_to_base_sql_on = conversation_history[-1]["content"]
            additional_context_info_for_sql = (
                f"CONTEXT: The user's literal input was \"{user_query}\", which is an affirmative response. "
                f"This means the user wants to proceed with the action or answer related to the assistant's previous statement (provided below as the main query). "
                f"Therefore, the PostgreSQL query you generate MUST address that previous statement."
            )
        else:
            additional_context_info_for_sql = "Generate a PostgreSQL aggregation query for the following financial question:"

        if conversation_history:
            formatted_history_for_sql_prompt += "\n--- Recent Conversation History (for SQL context) ---\n"
            for turn in conversation_history:
                formatted_history_for_sql_prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
            formatted_history_for_sql_prompt += "--- End of Recent Conversation History ---\n\n"

        # Get structured data via SQL
        sql_prompt = f"""
        {formatted_history_for_sql_prompt}
        {additional_context_info_for_sql}
        '''{query_to_base_sql_on}'''

        --- Database Schema and SQL Generation Rules (for hybrid context) ---
        Table: transactions (id, date, amount, merchant, category, description)
        The `category` field (e.g., 'GROCERIES', 'RESTAURANT') is the primary field for categorical queries.
        The `description` field contains more detailed text and can be used for keyword searches if category is not sufficient.
        Remember: Positive 'amount' values are income/deposits, negative 'amount' values are expenses/withdrawals.
        Focus on totals, averages, or counts. If the user asks about spending, filter for negative amounts (or sum absolute values of negative amounts if a positive total spending is needed). If they ask about income, filter for positive amounts.
        If a category is mentioned, use the `category` column for filtering (e.g., `WHERE category = 'ENTERTAINMENT'`).
        Return only SQL.
        """
        sql_query = self.query_ollama(sql_prompt).strip()
        sql_results = self.execute_sql(sql_query)
        
        # Get contextual data via vector search
        query_embedding = self.embed_model.encode([user_query])[0]
        vector_results = self.vector_search(query_embedding, limit=5)
        
        response_prompt = f"""
        Question: {user_query}
        
        Quantitative analysis (from SQL): {sql_results}
        Related transactions (from vector search): {vector_results}
        
        Remember: In the transaction data, positive amounts are income/deposits, and negative amounts are expenses/withdrawals.
        Combine the quantitative numbers with the context from related transactions to provide comprehensive financial advice. 
        Use appropriate financial terms like 'spent', 'earned', 'income', 'expenses' based on the data.
        Address the user's question directly and clearly.
        """
        
        return self.query_ollama(response_prompt)
    
    def direct_response(self, user_query: str) -> str:
        """Direct LLM response without RAG"""
        system_prompt = "You are a personal finance assistant. Provide general financial advice."
        return self.query_ollama(user_query, system_prompt)
    
    def process_query(self, user_query: str) -> str:
        """Main query processing with classification routing and conversation history management."""
        
        # Append user query to history
        self.conversation_history.append({"role": "user", "content": user_query})

        # Keep history to a reasonable length (e.g., last N turns) to avoid overly long prompts
        # For now, let's consider the last 5 turns (user + assistant pairs) for context, plus current user query
        # This means roughly 10 messages if a turn is a user query + assistant response.
        # The current user query is already added. We need up to 9 previous messages for 5 turns.
        max_history_messages_for_prompt = 9 
        history_for_prompt = self.conversation_history[-(max_history_messages_for_prompt + 1):-1] # Get history *before* current query

        intent = self.llm_classify_intent(user_query, conversation_history=history_for_prompt)
        print(f"[DEBUG] Classification: {intent}")
        
        response = ""
        if intent == 'sql':
            response = self.sql_rag(user_query, conversation_history=history_for_prompt)
        elif intent == 'vector':
            response = self.vector_rag(user_query, conversation_history=history_for_prompt)
        elif intent == 'hybrid':
            response = self.hybrid_rag(user_query, conversation_history=history_for_prompt)
        else:  # none
            response = self.direct_response(user_query)

        # Append assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim overall history if it gets too long (e.g., keep last 50 messages total)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
            
        return response

def main():
    parser = argparse.ArgumentParser(description="Financial ChatBot with RAG and Query Classification")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use for Ollama")
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