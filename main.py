from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dtos import SearchResponseDTO, SearchRequestDTO
from vehicle_search_rag import VehicleSearchRag
import os, io, uvicorn
from PIL import Image
from dotenv import load_dotenv


SECRET_PATH = None
load_dotenv(SECRET_PATH)


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL: str = "facebook/dinov2-base"
COMPRESSOR_MODEL_PATH: str = "compressor_model.pt"
ADDITIONAL_WEIGHTS_PATH: str = "additional_weights"
COLLECTION_NAME = "used cars dataset"


app = FastAPI()

templates = Jinja2Templates(directory="templates")

rag = VehicleSearchRag(
    QDRANT_API_KEY, QDRANT_URL, COLLECTION_NAME,
    model_name=EMBEDDING_MODEL,
    compressor_model_path=COMPRESSOR_MODEL_PATH,
    llm_model_path=ADDITIONAL_WEIGHTS_PATH
)


MAX_IMAGES_ALLOWED = 5


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request
    )


@app.post("/search/", response_model=SearchResponseDTO)
async def search_cars(
        search_params: SearchRequestDTO = Depends(SearchRequestDTO.as_form),
        files: list[UploadFile] = File(...)
) -> SearchResponseDTO:

    if len(files) > MAX_IMAGES_ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"to much files loaded"
        )

    pil_images = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        pil_images.append(image)

    return rag(pil_images, search_params)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)