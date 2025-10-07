from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

summarizer = pipeline("summarization", model="google/flan-t5-base")
app = FastAPI()

class ReviewsInput(BaseModel):
    reviews: List[str]

@app.post("/reviews")
def reviews(data: ReviewsInput):
    combined_text = " ".join(data.reviews)
    summary = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)
    return {"summary": summary[0]['summary_text']}