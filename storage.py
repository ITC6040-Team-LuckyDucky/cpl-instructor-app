import os

BLOB_CONTAINER = "cpl-uploads"
LOCAL_UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "local_uploads")


def _is_azure():
    return bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))


def ensure_storage_ready():
    """
    Creates the blob container (Azure) or local uploads folder (local dev)
    if it does not already exist.
    """
    if _is_azure():
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container = blob_service.get_container_client(BLOB_CONTAINER)
        try:
            container.create_container()
        except Exception:
            pass  # Already exists — not an error
    else:
        os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)


def download_file(blob_url):
    """
    Downloads and returns the raw bytes for a previously-uploaded file.

    Azure mode  → fetches from Blob Storage using the stored blob URL.
    Local mode  → reads from the local filesystem (blob_url starts with local://).
    """
    if blob_url.startswith("local://"):
        local_path = blob_url[len("local://"):]
        with open(local_path, "rb") as fh:
            return fh.read()

    # Azure Blob Storage
    from azure.storage.blob import BlobServiceClient
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container = blob_service.get_container_client(BLOB_CONTAINER)
    # Blob URL format: https://<account>.blob.core.windows.net/<container>/<blob_name>
    blob_name = blob_url.split(f"/{BLOB_CONTAINER}/", 1)[-1]
    blob_client = container.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def upload_file(file_bytes, filename, upload_id):
    """
    Stores file_bytes and returns a URL string.

    Azure mode  → uploads to Blob Storage, returns the blob HTTPS URL.
    Local mode  → saves to ./local_uploads/, returns a local:// path.
    """
    blob_name = f"{upload_id}_{filename}"

    if _is_azure():
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container = blob_service.get_container_client(BLOB_CONTAINER)
        try:
            container.create_container()
        except Exception:
            pass  # Already exists
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        return blob_client.url

    # Local dev fallback
    os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_UPLOADS_DIR, blob_name)
    with open(local_path, "wb") as fh:
        fh.write(file_bytes)
    return f"local://{local_path}"
