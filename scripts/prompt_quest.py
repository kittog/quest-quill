import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generer_quete(model, tokenizer, map, difficulty, target):
    prompt = f"Créer une quête sur la carte {map}, avec une difficulté {difficulty}, ciblant {target}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Charger le modèle entraîné
model_path = "model_entraîné"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Interface utilisateur pour la saisie des mots-clés
map = input("Entrez la carte : ")
difficulty = input("Entrez la difficulté (en chiffres) : ")
target = input("Entrez la cible : ")

# Génération de la quête
quete_generee = generer_quete(model, tokenizer, map, difficulty, target)
print("\nQuête Générée :\n", quete_generee)
