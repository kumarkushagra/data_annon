import asyncio
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field, ValidationError
from ollama import Client


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------

LABELS = ["plastic","paper_cardboard","metal","glass","organic_food","textile","rubber","wood","e_waste","hazardous",]

class GarbageScores(BaseModel):
    plastic: float = Field(ge=0.0, le=1.0)
    paper_cardboard: float = Field(ge=0.0, le=1.0)
    metal: float = Field(ge=0.0, le=1.0)
    glass: float = Field(ge=0.0, le=1.0)
    organic_food: float = Field(ge=0.0, le=1.0)
    textile: float = Field(ge=0.0, le=1.0)
    rubber: float = Field(ge=0.0, le=1.0)
    wood: float = Field(ge=0.0, le=1.0)
    e_waste: float = Field(ge=0.0, le=1.0)
    hazardous: float = Field(ge=0.0, le=1.0)

class Annotator:
    def __init__(self,ollama_host: str = "http://localhost:11434",model: str = "qwen3-vl",):
        self.client = Client(host=ollama_host)
        self.model = model

    # ---------------- PRIVATE ----------------

    def _build_prompt(self) -> str:
        return (
        """
            You are a waste classification system.

            For the image, assign confidence scores (0.0–1.0) for EXACTLY these 10 categories:

            plastic, paper_cardboard, metal, glass, organic_food,
            textile, rubber, wood, e_waste, hazardous

            RULES:
            - Output EXACTLY these 10 keys.
            - Values must be floats between 0.0 and 1.0.
            - Multi-label allowed.
            - For categories NOT clearly visible, use EXACTLY 0.0.
            - Do NOT hedge with small values (no 0.05, 0.1, etc).
            - Most images contain at most 1–2 categories.
            - At least 7 of the 10 values should usually be 0.0.
            - If one category is dominant, all others must be 0.0.
            - If no garbage is visible, set ALL values to 0.0.

            Return ONLY valid JSON. No text.
        """
    )

    async def _call_llm(self, image_path: str) -> str:
        """
        Async wrapper around ollama client.
        Returns raw JSON string.
        """
        return await asyncio.to_thread(
            lambda: self.client.chat(
                model=self.model,
                format=GarbageScores.model_json_schema(),
                messages=[
                    {
                        "role": "user",
                        "content": self._build_prompt(),
                        "images": [image_path],
                    }
                ],
                options={"temperature": 0},
            )["message"]["content"]
        )

    def _validate(self, raw_json: str) -> Dict[str, float]:
        try:
            parsed = GarbageScores.model_validate_json(raw_json)
        except ValidationError as e:
            raise ValueError(f"Invalid LLM output: {e}")

        return parsed.model_dump()

    # ---------------- PUBLIC ----------------

    async def annotate_image(self, image_path: str) -> Dict[str, float]:
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_json = await self._call_llm(str(path))
        return self._validate(raw_json)



if __name__ == "__main__":
    async def demo():
        annotator = Annotator()
        result = await annotator.annotate_image("/home/vector/dataset/master_dataset/raw/TrashNet/dataset-resized/trash/trash134.jpg")
        print(result)

    asyncio.run(demo())
