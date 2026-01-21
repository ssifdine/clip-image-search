import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os

class ImageSearchSystem:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialiser le système CLIP"""
        print("Chargement du modèle CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_embeddings = []
        self.image_paths = []
        print(f"Modèle chargé sur {self.device}")
    
    def encode_images(self, image_folder):
        """Encoder toutes les images d'un dossier"""
        print(f"Encodage des images dans {image_folder}...")
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        embedding = self.model.get_image_features(**inputs)
                    
                    # Normalisation
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    
                    self.image_embeddings.append(embedding.cpu().numpy())
                    self.image_paths.append(image_path)
                    
                except Exception as e:
                    print(f"Erreur avec {filename}: {e}")
        
        self.image_embeddings = np.vstack(self.image_embeddings)
        print(f"{len(self.image_paths)} images encodées!")
    
    def search(self, text_query, top_k=5):
        """Rechercher des images par description texte"""
        # Encoder la requête texte
        inputs = self.processor(text=text_query, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        
        # Normalisation
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy()
        
        # Calcul de similarité cosinus
        similarities = np.dot(self.image_embeddings, text_embedding.T).squeeze()
        
        # Top-K résultats
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'path': self.image_paths[idx],
                'score': float(similarities[idx])
            })
        
        return results

# Utilisation
if __name__ == "__main__":
    # Créer le système
    system = ImageSearchSystem()
    
    # Encoder le dataset d'images
    system.encode_images("./dataset_images")
    
    # Rechercher
    query = "un chat qui dort"
    results = system.search(query, top_k=5)
    
    print(f"\nRésultats pour '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['path']} - Score: {result['score']:.3f}")