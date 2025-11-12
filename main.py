from fastapi import FastAPI
from src.routes import reviews

app = FastAPI(title="Mi API Organizada", version="1.0.0")

# Registrar rutas
app.include_router(reviews.router, prefix="/api/reviews", tags=["Reviews"])

@app.get("/")
def root():
    return {"message": "Bienvenido a la API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
