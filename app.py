# Installation : pip install gradio pillow
import gradio as gr
from PIL import Image          # 👈 IMPORT MANQUANT
from image_search import ImageSearchSystem

# Initialiser le système
system = ImageSearchSystem()
system.encode_images("./dataset_images")

def search_images(query):
    """Fonction de recherche pour Gradio"""
    results = system.search(query, top_k=6)
    
    images = []
    for result in results:
        img = Image.open(result['path'])
        images.append((img, f"Score: {result['score']:.2%}"))
    
    return images

# Interface Gradio
interface = gr.Interface(
    fn=search_images,
    inputs=gr.Textbox(
        label="Description de l'image recherchée",
        placeholder="Ex: un chat qui dort, montagne enneigée..."
    ),
    outputs=gr.Gallery(label="Résultats", columns=3),
    title="🔍 Système de Recherche d'Images CLIP",
    description="Recherchez des images par description en langage naturel",
    examples=[
        ["un chat orange"],
        ["montagne avec neige"],
        ["personne en costume"],
        ["nourriture délicieuse"]
    ]
)

interface.launch(share=True)
