import asyncio
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field, ValidationError
from ollama import Client
import json


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
    def __init__(self,ollama_host: str = "http://localhost:11434",model: str = "gemma3",):
        self.client = Client(host=ollama_host)
        self.model = model

    # ---------------- PRIVATE ----------------

    def _build_prompt(self) -> str:
        return """
            You are a waste classification system.

            Analyze the image and return a JSON object with EXACTLY these 10 keys.
            Each value is a confidence score between 0.0 and 1.0.
            return a float for each key.
            
            Rules:
            - Scores must reflect actual presence in the image
            - Use 0.0 if absent
            - Higher score = more dominant material
            - No explanations, no extra text, ONLY JSON

            Keys:
            plastic, paper_cardboard, metal, glass,
            organic_food, textile, rubber, wood,
            e_waste, hazardous
            
            return a valid json 
        """

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
                        "role": "agent",
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
        result = await annotator.annotate_image(
            "/home/vector/dataset/master_dataset/raw/TrashNet/dataset-resized/trash/trash26.jpg"
        )
        print(json.dumps(result, indent=4, sort_keys=True))

    asyncio.run(demo())
