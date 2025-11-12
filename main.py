from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline
import torch

app = FastAPI(
    title="Review Generation API",
    description="Genera reseñas basadas en comentarios de productos"
)

# Configurar dispositivo
device = 0 if torch.cuda.is_available() else -1

# Cargar modelos (puedes cambiar según tus necesidades)
print("Cargando modelos...")

# Modelo para inglés (FLAN-T5)
generator_en = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device
)

# Modelo para español (mT5)
generator_es = pipeline(
    "text2text-generation", 
    model="google/mt5-base",
    device=device
)

print("Modelos cargados!")

# Modelos de datos
class ReviewGenerationRequest(BaseModel):
    comments: List[str] = Field(..., description="Lista de comentarios del producto")
    language: str = Field("en", description="Idioma: 'en' o 'es'")
    max_length: Optional[int] = Field(150, description="Longitud máxima de la reseña")
    min_length: Optional[int] = Field(40, description="Longitud mínima de la reseña")
    num_reviews: Optional[int] = Field(1, description="Número de reseñas a generar")
    temperature: Optional[float] = Field(0.8, description="Temperatura para generación")

class ReviewGenerationResponse(BaseModel):
    generated_reviews: List[str]
    input_comments: List[str]
    language: str
    model_used: str

@app.get("/")
def root():
    """Información de la API"""
    return {
        "message": "Review Generation API",
        "version": "1.0",
        "endpoints": {
            "/generate": "POST - Genera reseñas desde comentarios",
            "/generate-positive": "POST - Genera reseña positiva",
            "/generate-negative": "POST - Genera reseña negativa",
            "/models": "GET - Lista modelos disponibles",
            "/health": "GET - Estado de salud"
        }
    }

@app.get("/models")
def list_models():
    """Lista los modelos disponibles"""
    return {
        "english": {
            "name": "google/flan-t5-base",
            "size": "250M parameters",
            "description": "FLAN-T5 optimizado para tareas de texto"
        },
        "spanish": {
            "name": "google/mt5-base",
            "size": "580M parameters", 
            "description": "mT5 multilingüe con soporte para español"
        }
    }

@app.get("/health")
def health_check():
    """Verifica el estado de la API"""
    return {
        "status": "healthy",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "models_loaded": ["flan-t5-base", "mt5-base"]
    }

@app.post("/generate", response_model=ReviewGenerationResponse)
def generate_review(request: ReviewGenerationRequest):
    """
    Genera reseñas basadas en comentarios de productos.
    
    Ejemplo de uso:
    ```json
    {
        "comments": [
            "El producto llegó rápido",
            "Buena calidad",
            "Lo recomiendo"
        ],
        "language": "es",
        "max_length": 100,
        "num_reviews": 2
    }
    ```
    """
    
    try:
        # Seleccionar modelo según idioma
        generator = generator_es if request.language == "es" else generator_en
        model_name = "mt5-base" if request.language == "es" else "flan-t5-base"
        
        # Combinar comentarios
        combined_comments = " | ".join(request.comments)
        
        # Crear prompt según idioma
        if request.language == "es":
            prompt = f"Genera una reseña de producto basándote en estos comentarios: {combined_comments}"
        else:
            prompt = f"Generate a product review based on these comments: {combined_comments}"
        
        # Limitar número de reseñas
        num_reviews = min(request.num_reviews, 5)
        
        # Generar reseñas
        results = generator(
            prompt,
            max_length=request.max_length,
            min_length=request.min_length,
            num_return_sequences=num_reviews,
            temperature=request.temperature,
            do_sample=True if num_reviews > 1 else False,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Extraer textos generados
        generated_reviews = [result['generated_text'] for result in results]
        
        return ReviewGenerationResponse(
            generated_reviews=generated_reviews,
            input_comments=request.comments,
            language=request.language,
            model_used=model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reseña: {str(e)}")

@app.post("/generate-positive")
def generate_positive_review(
    product_name: str,
    key_features: List[str],
    language: str = "en",
    max_length: int = 100
):
    """
    Genera una reseña positiva para un producto.
    
    Ejemplo:
    ```json
    {
        "product_name": "Laptop HP",
        "key_features": ["rápida", "buena pantalla", "ligera"],
        "language": "es"
    }
    ```
    """
    
    generator = generator_es if language == "es" else generator_en
    
    features_text = ", ".join(key_features)
    
    if language == "es":
        prompt = f"Escribe una reseña positiva del producto {product_name} destacando: {features_text}"
    else:
        prompt = f"Write a positive review of {product_name} highlighting: {features_text}"
    
    result = generator(
        prompt,
        max_length=max_length,
        min_length=40,
        temperature=0.7,
        do_sample=True
    )
    
    return {
        "product": product_name,
        "review": result[0]['generated_text'],
        "sentiment": "positive",
        "language": language
    }

@app.post("/generate-negative")
def generate_negative_review(
    product_name: str,
    issues: List[str],
    language: str = "en",
    max_length: int = 100
):
    """
    Genera una reseña negativa para un producto.
    
    Ejemplo:
    ```json
    {
        "product_name": "Auriculares XYZ",
        "issues": ["mala calidad de sonido", "se rompieron rápido"],
        "language": "es"
    }
    ```
    """
    
    generator = generator_es if language == "es" else generator_en
    
    issues_text = ", ".join(issues)
    
    if language == "es":
        prompt = f"Escribe una reseña negativa del producto {product_name} mencionando estos problemas: {issues_text}"
    else:
        prompt = f"Write a negative review of {product_name} mentioning these issues: {issues_text}"
    
    result = generator(
        prompt,
        max_length=max_length,
        min_length=40,
        temperature=0.7,
        do_sample=True
    )
    
    return {
        "product": product_name,
        "review": result[0]['generated_text'],
        "sentiment": "negative",
        "language": language
    }

@app.post("/summarize-and-generate")
def summarize_and_generate(
    reviews: List[str],
    language: str = "en"
):
    """
    Resume múltiples reseñas y genera una nueva basada en el resumen.
    """
    
    generator = generator_es if language == "es" else generator_en
    
    # Paso 1: Resumir
    combined = " ".join(reviews)
    
    if language == "es":
        summary_prompt = f"Resume estos comentarios: {combined}"
    else:
        summary_prompt = f"Summarize these comments: {combined}"
    
    summary = generator(
        summary_prompt,
        max_length=100,
        min_length=30
    )
    
    # Paso 2: Generar reseña desde resumen
    summary_text = summary[0]['generated_text']
    
    if language == "es":
        generate_prompt = f"Escribe una reseña completa basada en: {summary_text}"
    else:
        generate_prompt = f"Write a complete review based on: {summary_text}"
    
    generated_review = generator(
        generate_prompt,
        max_length=150,
        min_length=50,
        temperature=0.8,
        do_sample=True
    )
    
    return {
        "original_reviews": reviews,
        "summary": summary_text,
        "generated_review": generated_review[0]['generated_text'],
        "language": language
    }

# Para ejecutar:
# uvicorn nombre_archivo:app --reload --host 0.0.0.0 --port 8000