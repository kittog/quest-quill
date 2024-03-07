from transformers import BertForMaskedLM, BertTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Chargement du tokenizer et du modèle pré-entraînés
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Charger et prétraiter les données pour BERT
def load_dataset(file_path, tokenizer):
    # un objet TextDataset pour traiter les données
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )
    # un objet DataCollatorForLanguageModeling pour le prétraitement des données
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    return dataset, data_collator

# Chargement des données d'entraînement
dataset, data_collator = load_dataset('quests_train.txt', tokenizer)

# Configuration des paramètres d'entraînement
training_args = TrainingArguments(
    output_dir="./bert-quests",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Création de l'objet Trainer pour l'entraînement du modèle
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Entraînement du modèle
trainer.train()

# Définition du répertoire de sauvegarde pour le modèle entraîné
output_model_dir = "./saved_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

# Importation de modules supplémentaires
import torch
from transformers import BertForMaskedLM, BertTokenizer

# Définition d'une fonction pour générer une quête en utilisant le modèle BERT pré-entraîné
def generer_quete(model, tokenizer, map, difficulty, target):
    # Construction du prompt pour la génération de la quête
    prompt = f"Créer une quête sur la carte {map}, avec une difficulté {difficulty}, ciblant {target}. La quête est : [CLS]"
    # Encodage du prompt en tensor
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Génération de la quête en utilisant le modèle
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1)
    # Décodage de la sortie pour obtenir la quête générée
    generated_quete = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_quete

# Chargement du modèle pré-entraîné pour la génération de quêtes
model_path = "bert"  
model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Demande d'informations à l'utilisateur pour générer une quête
map = input("Entrez la carte : ")
difficulty = input("Entrez la difficulté (en chiffres) : ")
target = input("Entrez la cible : ")

# Appel de la fonction de génération de quête et affichage du résultat
quete_generee = generer_quete(model, tokenizer, map, difficulty, target)
print("\nQuête Générée :\n", quete_generee)
