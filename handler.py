import runpod
import os
import json
import subprocess
import tempfile
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

DRIVE_FOLDER_ID = "11TRVCWM6HHMbxA0RJOHiULT5OuCzMABw"

def get_drive_service():
    sa_info = json.loads(os.environ["GOOGLE_SA_KEY"])
    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

DRIVE = get_drive_service()
print("Drive client ready.")

def download_from_drive(file_id, dest):
    request = DRIVE.files().get_media(fileId=file_id)
    with open(dest, "wb") as f:
        dl = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = dl.next_chunk()

def upload_to_drive(local_path, folder_id):
    media = MediaFileUpload(str(local_path), resumable=True)
    meta = {"name": local_path.name, "parents": [folder_id]}
    f = DRIVE.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]

def handler(job):
    inp = job["input"]
    source_file_id   = inp.get("source_file_id")
    hint_file_id     = inp.get("hint_file_id")
    output_folder_id = inp.get("output_folder_id", DRIVE_FOLDER_ID)
    device           = inp.get("device", "cuda")
    despill          = inp.get("despill_strength", 0)
    src_ext          = inp.get("source_ext", "png")
    hint_ext         = inp.get("hint_ext", "png")

    if not source_file_id or not hint_file_id:
        return {"error": "source_file_id and hint_file_id are required"}

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_dir = Path(tmpdir) / "clips"
        shot_dir  = clips_dir / "shot"
        out_dir   = Path(tmpdir) / "output"
        (shot_dir / "Input").mkdir(parents=True)
        (shot_dir / "AlphaHint").mkdir(parents=True)
        out_dir.mkdir()

        runpod.serverless.progress_update(job, "Downloading from Drive...")
        download_from_drive(source_file_id, shot_dir / "Input"     / f"frame.{src_ext}")
        download_from_drive(hint_file_id,   shot_dir / "AlphaHint" / f"frame.{hint_ext}")

        runpod.serverless.progress_update(job, "Running CorridorKey inference...")
        result = subprocess.run(
            [
                "/app/.venv/bin/python", "corridorkey_cli.py",
                "--action", "run_inference",
                "--clips_dir", str(clips_dir),
                "--output_dir", str(out_dir),
                "--device", device,
                "--despill_strength", str(despill),
            ],
            capture_output=True, text=True, cwd="/app",
        )

        if result.returncode != 0:
            return {"error": "Inference failed", "stderr": result.stderr[-3000:]}

        runpod.serverless.progress_update(job, "Uploading results to Drive...")
        uploaded = {}
        for folder in ["Processed", "Matte", "FG", "Comp"]:
            folder_path = out_dir / "shot" / folder
            if folder_path.exists():
                for f in sorted(folder_path.iterdir()):
                    file_id = upload_to_drive(f, output_folder_id)
                    uploaded[folder.lower()] = {
                        "filename": f.name,
                        "drive_file_id": file_id,
                        "drive_url": f"https://drive.google.com/file/d/{file_id}/view"
                    }

        return {"outputs": uploaded}

runpod.serverless.start({"handler": handler})
