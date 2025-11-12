class ReviewController:
    def __init__(self):
        # Simulación de base de datos
        self.reviews = [
            {"id": 1, "user": "Ron", "comment": "Great product!", "rating": 5},
            {"id": 2, "user": "Ana", "comment": "Not bad", "rating": 3}
        ]
    
    def get_all_reviews(self):
        return {"reviews": self.reviews}
    