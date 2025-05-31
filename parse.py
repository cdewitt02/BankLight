# import pdfplumber
import csv
from db import insertTransactionsBatch
import os

# def extract_text_from_pdf(pdf_path):
#     transactions = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 transactions.append(text)
#     return "\n".join(transactions)

# def frost_extract_transactions(pdf_text):
    
#     deposit = False
#     withdraw = False
    
#     deposits = []
#     withdraws = []
#     for line in pdf_text.split("\n"):
#         if "--------------------------------------- DEPOSITS/CREDITS -------------------------------------------" in line:
#             deposit = True
#             continue
#         if "------------------------------------- OTHER WITHDRAWALS/DEBITS -----------------------------------" in line:
#             deposit = False
#             withdraw = True
#             continue
#         if "Please examine your bank statement upon receipt" in line:
#             withdraw = False
#             continue
#         if "---------------------------------------- DAILY BALANCE ---------------------------------------------" in line:
#             return deposits[1:], withdraws[1:]
#         if deposit:
#             if line.strip():
#                 deposits.append(line)
#         if withdraw:    
#             if line.strip():
#                 withdraws.append(line)
                
#     return deposits[1:], withdraws[1:]

def extract_transactions_from_csv(csv_file, account):
    """
    Extract transactions from a CSV file and insert them into the database.
    Handles differences in column names for Amex and Frost accounts.
    """
    # Define column mappings for different account types
    column_mappings = {
        'amex': {'date': 'Date', 'description': 'Description', 'amount': 'Amount'},
        'frost': {'date': 'Transaction Date', 'description': 'Description', 'amount': 'Amount'},
        'fidelity': {'date': 'Date', 'description': 'Name', 'amount': 'Amount'}
    }

    # Get the correct column names for the given account type
    columns = column_mappings.get(account.lower())
    if not columns:
        print(f"Unknown account type: {account}")
        return

    with open(csv_file, 'r') as file:
        csvreader = csv.DictReader(file)  # Use DictReader to handle column names
        transactions = []
        for row in csvreader:
            # Extract the required fields using the mapped column names
            transaction_date = row.get(columns['date'], '').strip()
            description = row.get(columns['description'], '').strip()
            amount = row.get(columns['amount'], '').replace('$', '').replace(',', '').strip()  # Remove '$' and ','

            # Skip rows with missing data
            if not transaction_date or not description or not amount:
                print(f"Skipping row due to missing data: {row}")
                continue

            # Prepare the transaction as a dictionary
            transaction = {
                'date': transaction_date,
                'description': description,
                'amount': float(amount),
                'account': account
            }

            transactions.append(transaction)

        insertTransactionsBatch(transactions)

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
