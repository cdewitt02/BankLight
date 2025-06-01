# import pdfplumber
import csv
from db import insertTransactionsBatch
import os
import requests # Added for Ollama API calls

# Ollama API configuration (can be moved to a config file or env variables later)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b" # Using a default from model.py

ALLOWED_CATEGORIES = [
    "HOUSING/UTILITIES", 
    "GROCERIES", 
    "GAS", 
    "RESTAURANT", 
    "ENTERTAINMENT", 
    "MERCHANDISE & SUPPLIES",
    "TRANSPORTATION",
    "MISCELLANEOUS" 
]

def _query_ollama_for_category(description_text: str, model_name: str = DEFAULT_OLLAMA_MODEL) -> str:
    """
    Query the Ollama API to categorize the transaction based on its description.
    """
    categories_str = "\n- ".join(ALLOWED_CATEGORIES)
    prompt = f"""Analyze the following transaction description and categorize it into one of the listed categories.
Your goal is to accurately assign a category. Prioritize specific categories over MISCELLANEOUS if there's a reasonable indication.

Transaction Description:
'{description_text}'

Categories & Keyword Hints (use these to guide your choice):
- HOUSING/UTILITIES: (Keywords: Rent, Mortgage, Utilities, Electricity, Water, Gas (for home), Internet, Phone Bill, Insurance (home, renters), HOA, Property Tax)
- GROCERIES: (Keywords: Grocery, Supermarket, Market, Food Store, Whole Foods, Kroger, Safeway)
- GAS: (Keywords: Gas Station, Fuel, Chevron, Shell, Exxon, BP, Phillips 66, Conoco)
- RESTAURANT: (Keywords: Restaurant, Cafe, Diner, Bistro, Pub, Bar, Food Delivery, Grubhub, Uber Eats, Doordash, Starbucks, McDonalds, Pizza)
- ENTERTAINMENT: (Keywords: Movies, Concert, Theatre, Games, Spotify, Netflix, Hulu, Steam, Tickets, Museum, Park, Fitness, Gym, Yoga, Sports)
- MERCHANDISE & SUPPLIES: (Keywords: Shopping, Store, Online Store, Amazon, Walmart, Target, Best Buy, Clothing, Electronics, Books, Office Supplies, Pharmacy (for non-prescription items))
- TRANSPORTATION: (Keywords: Uber, Lyft, Taxi, Airline, Train, Bus, Public Transit, Parking, Tolls, Subway, Ride Share, Airline Tickets, Baggage Fees)
- MISCELLANEOUS: (Use this if no other category strongly fits. Examples: Bank Fees, ATM Withdrawal, Unclear Purchases, Gifts if category unknown)

Your response MUST be exactly one of the category names from the list (e.g., HOUSING/UTILITIES, GROCERIES, etc.).
Do not add any other text, explanation, or punctuation.

Chosen Category:
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=10) # Shorter timeout for category
        if response.status_code == 200:
            category = response.json().get("response", "").strip().upper() # Ensure uppercase for comparison
            if category in ALLOWED_CATEGORIES:
                print(f"[LLM Category] Description: '{description_text}' -> Category: '{category}'")
                return category
            else:
                print(f"[LLM Category] LLM returned non-allowed category: '{category}'. Defaulting to MISCELLANEOUS.")
        else:
            print(f"[LLM Error] Failed to categorize description '{description_text}'. Status: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[LLM Error] Request failed for category assignment of '{description_text}': {e}")
    
    print(f"[LLM Category] Defaulting to MISCELLANEOUS for '{description_text}'")
    return "MISCELLANEOUS" # Fallback category

def extract_transactions_from_csv(csv_file, account):
    """
    Extract transactions from a CSV file and insert them into the database.
    Handles differences in column names for Amex and Frost accounts.
    Skips specified transactions and categorizes them using an LLM based on original description.
    """
    column_mappings = {
        'amex': {'date': 'Date', 'description': 'Description', 'amount': 'Amount'},
        'frost': {'date': 'Transaction Date', 'description': 'Description', 'amount': 'Amount'},
        'fidelity': {'date': 'Date', 'description': 'Name', 'amount': 'Amount'}
    }

    columns = column_mappings.get(account.lower())
    if not columns:
        print(f"Unknown account type: {account}")
        return

    with open(csv_file, 'r') as file:
        csvreader = csv.DictReader(file)
        transactions = []
        for row in csvreader:
            transaction_date = row.get(columns['date'], '').strip()
            original_description = row.get(columns['description'], '').strip()
            amount_str = row.get(columns['amount'], '').replace('$', '').replace(',', '').strip()

            if not transaction_date or not original_description or not amount_str:
                print(f"Skipping row due to missing critical data (date, description, or amount): {row}")
                continue
            
            original_description_upper = original_description.upper()
            account_lower = account.lower()

            if "AMEX" in original_description_upper:
                print(f"Skipping transaction with 'AMEX' in description: '{original_description}'")
                continue
            if account_lower == 'frost' and "CARDMASTER" in original_description_upper:
                print(f"Skipping Frost transaction with 'CARDMASTER': '{original_description}'")
                continue
            if account_lower == 'fidelity' and "INTERNET PAYMENT" in original_description_upper:
                print(f"Skipping Fidelity transaction with 'INTERNET PAYMENT': '{original_description}'")
                continue
            if account_lower == 'amex' and "MOBILE-PAYMENT" in original_description_upper:
                print(f"Skipping Amex transaction with 'MOBILE-PAYMENT': '{original_description}'")
                continue
            
            # --- Categorization using original description ---
            category = _query_ollama_for_category(original_description)

            transaction = {
                'date': transaction_date,
                'description': original_description, # Using original_description
                'amount': float(amount_str),
                'account': account,
                'category': category
            }

            transactions.append(transaction)

        if transactions:
            insertTransactionsBatch(transactions)
        else:
            print(f"No transactions to insert for {csv_file} after filtering.")

def parse(file_path):
    """
    Parse the given file and process transactions based on its type.
    """
    if file_path.endswith('.csv'):
        # Determine the account type based on the file name
        if 'amex' in file_path.lower():
            extract_transactions_from_csv(file_path, 'amex')
        elif 'frost' in file_path.lower():
            extract_transactions_from_csv(file_path, 'frost')
        elif 'fidelity' in file_path.lower():
            extract_transactions_from_csv(file_path, 'fidelity')
        else:
            print("Unknown account type for CSV file.")
    else:
        print("Unsupported file format.")
        
    deleteDownload(file_path)
        

    
def deleteDownload(file_path):
    """
    Delete the downloaded file after processing.
    """
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")
