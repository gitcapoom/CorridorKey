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

def run_cmd(cmd, cwd="/app"):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])
    return result

def handler(job):
    inp = job["input"]

    source_file_id   = inp.get("source_file_id")        # Drive file ID of green screen clip/frame
    hint_file_id     = inp.get("hint_file_id")           # Optional: provide your own alpha hint
    output_folder_id = inp.get("output_folder_id", DRIVE_FOLDER_ID)
    hint_method      = inp.get("hint_method", "gvm")     # "gvm", "videomama", or "manual"
    device           = inp.get("device", "cuda")
    despill          = inp.get("despill_strength", 0)
    src_ext          = inp.get("source_ext", "mp4")

    if not source_file_id:
        return {"error": "source_file_id is required"}
    if hint_method == "manual" and not hint_file_id:
        return {"error": "hint_file_id is required when hint_method is manual"}

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_dir = Path(tmpdir) / "clips"
        shot_dir  = clips_dir / "shot"
        out_dir   = Path(tmpdir) / "output"
        input_dir  = shot_dir / "Input"
        hint_dir   = shot_dir / "AlphaHint"
        input_dir.mkdir(parents=True)
        hint_dir.mkdir(parents=True)
        out_dir.mkdir()

        # Download source footage from Drive
        runpod.serverless.progress_update(job, "Downloading source from Drive...")
        src_path = input_dir / f"source.{src_ext}"
        download_from_drive(source_file_id, src_path)

        # Generate or download alpha hint
        if hint_method == "manual":
            runpod.serverless.progress_update(job, "Downloading alpha hint from Drive...")
            download_from_drive(hint_file_id, hint_dir / f"source.png")

        elif hint_method == "gvm":
            runpod.serverless.progress_update(job, "Generating alpha hint with GVM...")
            run_cmd([
                "/app/.venv/bin/python", "clip_manager.py",
                "--action", "generate_hints",
                "--clips_dir", str(clips_dir),
                "--hint_method", "gvm",
                "--device", device,
            ])

        elif hint_method == "videomama":
            runpod.serverless.progress_update(job, "Generating alpha hint with VideoMaMa...")
            # VideoMaMa requires a rough mask hint — user must supply it
            if not hint_file_id:
                return {"error": "VideoMaMa requires a hint_file_id (rough mask) in the VideoMamaMaskHint folder"}
            mask_dir = shot_dir / "VideoMamaMaskHint"
            mask_dir.mkdir(parents=True)
            download_from_drive(hint_file_id, mask_dir / f"source.png")
            run_cmd([
                "/app/.venv/bin/python", "clip_manager.py",
                "--action", "generate_hints",
                "--clips_dir", str(clips_dir),
                "--hint_method", "videomama",
                "--device", device,
            ])

        # Run CorridorKey inference
        runpod.serverless.progress_update(job, "Running CorridorKey inference...")
        run_cmd([
            "/app/.venv/bin/python", "corridorkey_cli.py",
            "--action", "run_inference",
            "--clips_dir", str(clips_dir),
            "--output_dir", str(out_dir),
            "--device", device,
            "--despill_strength", str(despill),
        ])

        # Upload outputs to Drive
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
