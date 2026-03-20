import runpod
import os
import json
import subprocess
import tempfile
import zipfile
import shutil
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

DRIVE_FOLDER_ID = "11TRVCWM6HHMbxA0RJOHiULT5OuCzMABw"
APP_DIR = Path("/app")
# clip_manager.py uses hardcoded ClipsForInference and Output dirs relative to /app
CLIPS_DIR = APP_DIR / "ClipsForInference"
OUTPUT_DIR = APP_DIR / "Output"

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

def unzip_to(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(folder_path).iterdir()):
            z.write(f, f.name)

def handler(job):
    inp = job["input"]

    source_file_id   = inp.get("source_file_id")
    hint_file_id     = inp.get("hint_file_id")
    output_folder_id = inp.get("output_folder_id", DRIVE_FOLDER_ID)
    hint_method      = inp.get("hint_method", "gvm")
    device           = inp.get("device", "cuda")
    despill          = inp.get("despill_strength", 0)
    gamma            = inp.get("gamma", "linear")
    sequence_mode    = inp.get("sequence_mode", True)
    src_ext          = inp.get("source_ext", "zip")
    hint_ext         = inp.get("hint_ext", "zip")

    if not source_file_id:
        return {"error": "source_file_id is required"}
    if hint_method == "manual" and not hint_file_id:
        return {"error": "hint_file_id required for manual hint_method"}

    # Use a fixed shot name so clip_manager finds it under ClipsForInference/shot/
    shot_dir  = CLIPS_DIR / "shot"
    input_dir = shot_dir / "Input"
    hint_dir  = shot_dir / "AlphaHint"

    # Clean up from any previous run
    if shot_dir.exists():
        shutil.rmtree(shot_dir)
    input_dir.mkdir(parents=True)
    hint_dir.mkdir(parents=True)

    # Also clean Output dir
    shot_out = OUTPUT_DIR / "shot"
    if shot_out.exists():
        shutil.rmtree(shot_out)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download source
        runpod.serverless.progress_update(job, "Downloading source from Drive...")
        src_zip = tmpdir / f"source.{src_ext}"
        download_from_drive(source_file_id, src_zip)
        if sequence_mode and src_ext == "zip":
            unzip_to(src_zip, input_dir)
        else:
            shutil.copy(src_zip, input_dir / src_zip.name)

        # Handle hint
        if hint_method == "manual":
            runpod.serverless.progress_update(job, "Downloading hint from Drive...")
            hint_zip = tmpdir / f"hint.{hint_ext}"
            download_from_drive(hint_file_id, hint_zip)
            if sequence_mode and hint_ext == "zip":
                unzip_to(hint_zip, hint_dir)
            else:
                shutil.copy(hint_zip, hint_dir / hint_zip.name)

        elif hint_method == "gvm":
            runpod.serverless.progress_update(job, "Generating alpha hints with GVM...")
            run_cmd([
                "/app/.venv/bin/python", "clip_manager.py",
                "--action", "generate_alphas",
                "--device", device,
            ])

        elif hint_method == "videomama":
            runpod.serverless.progress_update(job, "Generating alpha hints with VideoMaMa...")
            if not hint_file_id:
                return {"error": "VideoMaMa requires hint_file_id (rough mask)"}
            mask_dir = shot_dir / "VideoMamaMaskHint"
            mask_dir.mkdir(parents=True)
            hint_zip = tmpdir / f"mask.{hint_ext}"
            download_from_drive(hint_file_id, hint_zip)
            if sequence_mode and hint_ext == "zip":
                unzip_to(hint_zip, mask_dir)
            else:
                shutil.copy(hint_zip, mask_dir / hint_zip.name)
            run_cmd([
                "/app/.venv/bin/python", "clip_manager.py",
                "--action", "generate_alphas",
                "--device", device,
            ])

        # Run inference — clip_manager writes output to Output/shot/
        runpod.serverless.progress_update(job, "Running CorridorKey inference...")
        run_cmd([
            "/app/.venv/bin/python", "clip_manager.py",
            "--action", "run_inference",
            "--device", device,
        ])

        # Zip and upload each output pass from Output/shot/
        runpod.serverless.progress_update(job, "Uploading results to Drive...")
        uploaded = {}
        out_shot = OUTPUT_DIR / "shot" / "Output"
        for folder in ["Processed", "Matte", "FG", "Comp"]:
            folder_path = out_shot / folder
            if folder_path.exists() and any(folder_path.iterdir()):
                zip_path = tmpdir / f"{folder}.zip"
                zip_folder(folder_path, zip_path)
                file_id = upload_to_drive(zip_path, output_folder_id)
                uploaded[folder.lower()] = {
                    "filename": f"{folder}.zip",
                    "drive_file_id": file_id,
                    "drive_url": f"https://drive.google.com/file/d/{file_id}/view"
                }

        return {"outputs": uploaded}

runpod.serverless.start({"handler": handler})
