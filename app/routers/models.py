from fastapi import APIRouter
from app.services.llm import FALLBACK_CHAIN

router = APIRouter()

@router.get("/")
def list_models():
    # Return the list of available models (fallback chain) along with id and label for frontend dropdown    
    return {
        "models": [
            {"id": m, "label": m.split(":")[0]} for m in FALLBACK_CHAIN
        ]
    }   
    

