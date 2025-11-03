# libryries 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline
# For dataset f
from datasets import load_dataset
import requests
import json

"""
# models
# "google/flane-t5-bas"  
# "facebook/bart-large-cnn", modelo mas simple, rquiere acceso
# "facebook/mbart-large-50", modelo multilingüe, mas pesado, requiere acceso 
# Modelos de traducción-> "Helsinki-NLP/opus-mt-en-es", con pipeline.
# Modelos mas potentes -> 
# LLaMA (Meta)
#from transformers import AutoModelForCausalLM, AutoTokenizer
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# Mistral (muy eficiente)
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Modelos en epañol 
"flax-community/t5-base-spanish", modelo descontinuado, error en la carga del modelo
"""
#summarizer = pipeline("summarization", model=  "flax-community/t5-base-spanish"  )
summarizer = pipeline("text2text-generation", model="google/mt5-base")  
app = FastAPI() 


class ReviewsInput(BaseModel):
    reviews: List[str]

@app.post("/reviews")
def reviews(data: ReviewsInput):
    combined_text = " ".join(data.reviews)
    summary = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)
    return {"summary": summary[0]['summary_text']}