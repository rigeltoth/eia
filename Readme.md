# FastAPI Starter

This small project provides a FastAPI app that summarizes a batch of product reviews using a Hugging Face transformers model.

## Endpoints

- GET /ping → health check that returns {"message": "pong"}
- POST /reviews or /reviews/ → Accept JSON body {"reviews": ["r1", "r2", ...]} and returns a single summary string.

Example POST payload:

```json
{
  "reviews": [
    "El producto me encantó, muy buena calidad",
    "La entrega fue rápida"
  ]
}
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/reviews" \
	-H "Content-Type: application/json" \
	-d '{"reviews": ["El producto me encantó, muy buena calidad", "La entrega fue rápida"]}'
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Recommended dev starter (use this to run the FastAPI dev server)

```bash
python main.py
```

Open docs at: http://127.0.0.1:8000/docs

Save current deps:

```bash
pip freeze > requirements.txt
```

.gitignore (if venv inside repo)
.venv/
venv/
eia/
**pycache**/
\*.py[cod]
