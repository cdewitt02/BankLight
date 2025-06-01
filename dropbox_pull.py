from flask import Flask, request
import os
import dropbox
from parse import parse
from dotenv import load_dotenv


app = Flask(__name__)

# --- Configuration for Processed Files Log ---
PROCESSED_FILES_LOG = "processed_files.log"

# Global Dropbox client instance, will be initialized in main
dbx = None

def is_file_processed(dropbox_file_path_lower: str) -> bool:
    """Checks if the Dropbox file path has already been processed.
    Reads from PROCESSED_FILES_LOG.
    """
    if not os.path.exists(PROCESSED_FILES_LOG):
        return False
    try:
        with open(PROCESSED_FILES_LOG, 'r') as f:
            processed_files = {line.strip() for line in f}
        return dropbox_file_path_lower in processed_files
    except Exception as e:
        print(f"Error reading processed files log '{PROCESSED_FILES_LOG}': {e}")
        return False # Be cautious; better to re-process than to miss a file due to log error

def mark_file_as_processed(dropbox_file_path_lower: str):
    """Marks the Dropbox file path as processed by appending to PROCESSED_FILES_LOG."""
    try:
        with open(PROCESSED_FILES_LOG, 'a') as f:
            f.write(f"{dropbox_file_path_lower}\n")
        print(f"Marked as processed: {dropbox_file_path_lower}")
    except Exception as e:
        print(f"Error writing to processed files log '{PROCESSED_FILES_LOG}': {e}")

def download_file_from_dropbox(file_path):
    """Downloads a file from Dropbox and calls the parse function.
    Does NOT mark the file as processed here; that's done by the caller after success.
    file_path: The Dropbox path_lower of the file.
    """
    local_filename = "downloads/" + file_path.split("/")[-1]
    # Ensure downloads directory exists
    os.makedirs("downloads", exist_ok=True)
    with open(local_filename, "wb") as f:
        metadata, res = dbx.files_download(file_path)
        f.write(res.content)
    print(f"Downloaded: {local_filename}")
    parse(local_filename) # This calls deleteDownload internally

@app.route("/dropbox-webhook", methods=["GET","POST"])
def dropbox_webhook():
    if request.method == "GET":
        challenge = request.args.get("challenge")
        return challenge, 200
    if request.method == "POST":
        return handle_dropbox_event(request)
    return "", 200


def handle_dropbox_event(request):
    data = request.get_json()
    print("Dropbox Event:", data)
    accounts = data.get("list_folder", {}).get("accounts", [])
    if accounts:
        list_folder_result = dbx.files_list_folder("", recursive=False)
        entries = list_folder_result.entries
        if entries:
            last_entry = entries[-1]
            if isinstance(last_entry, dropbox.files.FileMetadata):
                dropbox_file_path = last_entry.path_lower
                print(f"Detected file: {dropbox_file_path}")
                if not is_file_processed(dropbox_file_path):
                    try:
                        print(f"Processing new file: {dropbox_file_path}")
                        download_file_from_dropbox(dropbox_file_path)
                        mark_file_as_processed(dropbox_file_path)
                    except Exception as e:
                        print(f"Error processing file {dropbox_file_path}: {e}. Will not mark as processed.")
                else:
                    print(f"Skipping already processed file: {dropbox_file_path}")
    return "", 200

if __name__ == "__main__":
    load_dotenv()
    ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    app.run(port=5000)
