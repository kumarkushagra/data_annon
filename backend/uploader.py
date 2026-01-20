import asyncio
import os
import hashlib
import aiofiles
from supabase import create_client, Client
from httpx import ConnectError
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")


class FileUploader:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.duplicates_log = "duplicates.log"
        self.semaphore = asyncio.Semaphore(4)  # kong-safe
        self.pbar = None  # tqdm progress bar

    async def calculate_hash(self, file_path: str) -> str:
        h = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    async def hash_exists(self, file_hash: str) -> bool:
        try:
            return await asyncio.to_thread(
                lambda: bool(
                    self.supabase
                    .table("images")
                    .select("id")
                    .eq("hash", file_hash)
                    .limit(1)
                    .execute()
                    .data
                )
            )
        except ConnectError:
            await asyncio.sleep(1)
            return False

    async def insert_image(self, file_path: str, file_hash: str):
        try:
            await asyncio.to_thread(
                lambda: self.supabase
                .table("images")
                .insert(
                    {"image_path": file_path, "hash": file_hash},
                    returning="minimal"
                )
                .execute()
            )
        except Exception:
            await self.log_duplicate(file_path)

    async def log_duplicate(self, file_path: str):
        async with aiofiles.open(self.duplicates_log, "a") as f:
            await f.write(file_path + "\n")

    async def handle_file(self, file_path: str):
        async with self.semaphore:
            try:
                file_hash = await self.calculate_hash(file_path)

                if await self.hash_exists(file_hash):
                    await self.log_duplicate(file_path)
                else:
                    await self.insert_image(file_path, file_hash)
            finally:
                # ALWAYS update progress (success / duplicate / error)
                self.pbar.update(1)

    async def process_directory(self, directory_path: str):
        image_paths = []

        for root, _, files in os.walk(directory_path):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, name))

        self.pbar = tqdm(
            total=len(image_paths),
            desc="Uploading images",
            unit="img"
        )

        tasks = [self.handle_file(p) for p in image_paths]
        await asyncio.gather(*tasks)

        self.pbar.close()

    # ------------------------------------------------------------------
    # LABEL UPDATE METHOD (FOR LATER USE)
    # ------------------------------------------------------------------
    async def update_image_labels(self, image_id: int, one_hot: dict):
        """
        Update labels for a given image.

        Parameters
        ----------
        image_id : int
            Primary key from `images.id`

        one_hot : dict
            One-hot encoded labels.
            Keys MUST match column names in `image_labels` table.
            Example:
            {
                "plastic": True,
                "metal": True,
                "glass": False,
                "organic_food": False,
                ...
            }
        """

        await asyncio.to_thread(
            lambda: self.supabase
            .table("image_labels")
            .update(one_hot)
            .eq("image_id", image_id)
            .execute()
        )


# ---------------- RUN ----------------

SUPABASE_URL = "http://127.0.0.1:8000"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"

if __name__ == "__main__":
    uploader = FileUploader(SUPABASE_URL, SUPABASE_KEY)
    asyncio.run(
        uploader.process_directory(
            "/home/vector/dataset/master_dataset/raw"
        )
    )
