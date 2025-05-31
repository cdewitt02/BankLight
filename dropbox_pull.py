from flask import Flask, request
import os
import dropbox
from parse import parse
from dotenv import load_dotenv


app = Flask(__name__)


def download_file_from_dropbox(file_path):
    local_filename = "downloads/" + file_path.split("/")[-1]
    with open(local_filename, "wb") as f:
        metadata, res = dbx.files_download(file_path)
        f.write(res.content)
    print(f"Downloaded: {local_filename}")
    parse(local_filename)

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
        entries = dbx.files_list_folder("").entries
        if entries:
            last_entry = entries[-1]
            if isinstance(last_entry, dropbox.files.FileMetadata):
                print(f"New File: {last_entry.path_lower}")
                download_file_from_dropbox(last_entry.path_lower)
    return "", 200

if __name__ == "__main__":
    load_dotenv()
    ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    app.run(port=5000)
