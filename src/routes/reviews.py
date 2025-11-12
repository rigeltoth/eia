from fastapi import APIRouter
from src.controllers.review_controller import ReviewController

router = APIRouter()
controller = ReviewController()

@router.get("/")
def get_reviews():
    return controller.get_all_reviews()