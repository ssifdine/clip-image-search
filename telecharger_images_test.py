# telecharger_images_test.py
import urllib.request
import os

def download_test_dataset():
    """Télécharger un petit dataset de test"""
    
    # URLs d'images gratuites (Unsplash)
    images = {
        "chat_1.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
        "chat_2.jpg": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800",
        "chat_3.jpg": "https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=800",
        "chien_1.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
        "chien_2.jpg": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",
        "chien_3.jpg": "https://images.unsplash.com/photo-1534351450181-ea9f78427fe8?w=800",
        "montagne_1.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "montagne_2.jpg": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800",
        "montagne_3.jpg": "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=800",
        "nourriture_1.jpg": "https://images.unsplash.com/photo-1551782450-a2132b4ba21d?w=800",
        "nourriture_2.jpg": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800",
        "nourriture_3.jpg": "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=800",
        "voiture_1.jpg": "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?w=800",
        "voiture_2.jpg": "https://images.unsplash.com/photo-1542362567-b07e54358753?w=800",
        "personne_1.jpg": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        "personne_2.jpg": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800",
        "livre_1.jpg": "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=800",
        "plage_1.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
        "ville_1.jpg": "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800",
        "fleur_1.jpg": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=800",
        "oiseau_1.jpg": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800",
        "cafe_1.jpg": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800",
        "ordinateur_1.jpg": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=800",
        "velo_1.jpg": "https://images.unsplash.com/photo-1571068316344-75bc76f77890?w=800",
        "nature_1.jpg": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    }
    
    os.makedirs("dataset_images", exist_ok=True)
    
    print("=" * 60)
    print("📥 TÉLÉCHARGEMENT DU DATASET DE TEST")
    print("=" * 60)
    print(f"🎯 {len(images)} images à télécharger\n")
    
    success_count = 0
    
    for i, (filename, url) in enumerate(images.items(), 1):
        try:
            filepath = os.path.join("dataset_images", filename)
            
            # Télécharger avec un user agent pour éviter les blocages
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
            
            print(f"[{i}/{len(images)}] ✅ {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(images)}] ❌ {filename}: {str(e)[:50]}")
    
    print("\n" + "=" * 60)
    print(f"✨ Téléchargement terminé!")
    print(f"✅ {success_count}/{len(images)} images téléchargées avec succès")
    print(f"📂 Images sauvegardées dans: ./dataset_images/")
    print("=" * 60)

if __name__ == "__main__":
    download_test_dataset()