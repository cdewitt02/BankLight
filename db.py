import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv() 

def get_db_credentials():
    """Get database credentials with proper error handling"""
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER') 
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    
    # Check if any required variable is missing
    required_vars = {
        'DB_NAME': db_name,
        'DB_USER': db_user, 
        'DB_PASSWORD': db_password,
        'DB_HOST': db_host,
        'DB_PORT': db_port
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
    return {
        'database': db_name, 
        'user': db_user,
        'password': db_password,
        'host': db_host,
        'port': db_port
    }

def create_embedding_text(transaction):
    """Create rich text for better semantic search from CSV-loaded transactions"""
    try:
        amount = float(transaction['amount'])
        amount_str = f"${amount:.2f}"
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid or missing 'amount': {transaction.get('amount')}") from e

    description = transaction.get('description', 'No description')
    account = transaction.get('account', 'Unknown account')

    return f"Amount: {amount_str} | Account: {account} | Description: {description}"


def get_db_connection():
    """Get database connection with proper error handling"""
    creds = get_db_credentials()
    try:
        conn = psycopg2.connect(**creds)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        raise

def insertTransaction(transaction):
    """
    Parse a transaction string into a dictionary and insert it into the database with embedding.
    """
    embedding_text = create_embedding_text(transaction)
    embedding = embed_model.encode(embedding_text)
    
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO transactions (date, description, amount, account, embedding)
            VALUES (%s, %s, %s, %s, %s)""",
            (
                transaction['date'],
                transaction['description'],
                transaction['amount'],
                transaction['account'],
                embedding.tolist() 
            )
        )
        conn.commit()
        print(f"Inserted transaction with embedding: {transaction}")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def insertTransactionsBatch(transactions):
    """
    Insert multiple transactions with embeddings in a single batch.
    """
    embedding_texts = [create_embedding_text(t) for t in transactions]
    embeddings = embed_model.encode(embedding_texts)
    
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        insert_data = [
            (t['date'], t['description'], t['amount'], t['account'], emb.tolist())
            for t, emb in zip(transactions, embeddings)
        ]
        
        cursor.executemany(
            """INSERT INTO transactions (date, description, amount, account, embedding)
            VALUES (%s, %s, %s, %s, %s)""",
            insert_data
        )
        conn.commit()
        print(f"Inserted {len(transactions)} transactions with embeddings")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()