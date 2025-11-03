

from datasets import load_dataset

# Este siempre funciona
dataset = load_dataset("yelp_review_full", split="train[:100]")

# Ver la estructura (esto es importante)
print("Columnas disponibles:", dataset.column_names)
print("\nPrimer elemento:")
print(dataset[1])
print("\nTipo:", type(dataset[1]))
