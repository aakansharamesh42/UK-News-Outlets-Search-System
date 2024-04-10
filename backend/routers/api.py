from fastapi import APIRouter
from .search import router as search_router
from .file import router as file_router
router = APIRouter()
router.include_router(search_router)
router.include_router(file_router)
