import asyncio
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
from uploader import FileUploader
from annotate import Annotator

SUPABASE_URL = "http://127.0.0.1:8000"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"

app = FastAPI()
uploader = FileUploader(SUPABASE_URL, SUPABASE_KEY)
annotator = Annotator()


@app.get("/health")
async def health():
    return {"status": "ok"}


async def run_upload(directory_path: str):
    await uploader.process_directory(directory_path)


@app.get("/upload")
async def upload(directory_path: str, background_tasks: BackgroundTasks):
    """
    Trigger upload from browser.

    Example:
    http://127.0.0.1:9000/upload?directory_path=/home/vector/dataset/master_dataset/raw
    """
    if not directory_path:
        raise HTTPException(status_code=400, detail="directory_path required")

    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="directory does not exist")

    background_tasks.add_task(run_upload, directory_path)
    return {"status": "started", "directory": directory_path}


class LabelRequest(BaseModel):
    image_id: int
    image_path: str


@app.post("/label")
async def label_image(data: LabelRequest):
    try:
        scores = await annotator.annotate_image(data.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        await asyncio.to_thread(
            lambda: uploader.supabase
            .table("image_labels")
            .update(scores)
            .eq("image_id", data.image_id)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB update failed: {e}")

    return {"status": "ok", "labels": scores}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
