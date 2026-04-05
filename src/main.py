"""main.py"""

import json
import shutil
import os
import uuid
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()
UPLOAD_DIR = "/data/uploads"

# In-memory store for job statuses
jobs = {}


@app.get("/download/{filename}")
async def download_censored_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=filename)


@app.post("/upload")
async def upload_media(background_tasks: BackgroundTasks, file: UploadFile = File(...), debug: bool = False):
    job_id = str(uuid.uuid4())[:8]  # Short unique ID
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    # Define what the output filename will look like
    output_filename = f"censored_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    # Save the uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize job status
    # jobs[job_id] = {"status": "processing", "file": output_filename}

    # newer version with more information on status
    jobs[job_id] = {
        "status": "processing",
        "progress_percent": 0,  # 0 to 100
        "eta_seconds": 0,  # Seconds remaining
        "current_step": "Initializing",
    }

    # Start the background task
    background_tasks.add_task(
        run_foul_filter_with_status, job_id, input_path, output_path, debug
    )

    return {"job_id": job_id, "message": "Upload successful. Processing started."}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status_file = f"/app/data/jobs/{job_id}.json"
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return json.load(f)

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs[job_id]


def run_foul_filter_with_status(job_id: str, input_p: str, output_p: str, debug: bool):
    base_name = os.path.splitext(os.path.basename(input_p))[0]
    tmp_dir = os.path.join("/app", f"{base_name}_tmp")

    try:
        import subprocess

        # 1. Execute the filter (saves to /app by default)
        cmd = ["python", "find_and_remove.py", "--job_id", job_id, input_p, "bad_words.txt"]
        
        if debug:
            cmd.append("--debug") # Pass flag to CLI

        subprocess.run(cmd, check=True)

        if debug:
            transcript_name = f"{os.path.splitext(os.path.basename(input_p))[0]}_transcript.txt"
            if os.path.exists(transcript_name):
                shutil.move(transcript_name, os.path.join(UPLOAD_DIR, transcript_name))
                jobs[job_id]["transcript_url"] = f"/download/{transcript_name}"

        # subprocess.run(
        #     [
        #         "python",
        #         "find_and_remove.py",
        #         "--job_id",
        #         job_id,
        #         input_p,
        #         "bad_words.txt",
        #     ],
        #     check=True,
        # )

        # 2. Identify the file that was just created in /app
        base = os.path.basename(input_p)
        name, ext = os.path.splitext(base)
        expected_output = f"{name}_CENSORED{ext}"

        source_path = os.path.join("/app", expected_output)
        destination_path = os.path.join(UPLOAD_DIR, expected_output)

        # 3. Move it to the mapped data volume
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["file"] = expected_output
            jobs[job_id]["download_url"] = f"/download/{expected_output}"
        else:
            raise FileNotFoundError(
                f"AI finished but {expected_output} was not found in /app"
            )

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # 3. Clean up the temp folder
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temporary directory: {tmp_dir}")


# def run_foul_filter(path: str):
#     # This calls your current find_and_remove.py
#     import subprocess
#     subprocess.run(["python", "find_and_remove.py", path, "bad_words.txt"])
