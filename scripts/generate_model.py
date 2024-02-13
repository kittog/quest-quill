from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "gpt2_quest_generator_finetuned2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Exemple d'entrée
prompt = "Map: Shrine Ruins | Difficulty: easy | Target: Rathian ->"

inputs = tokenizer.encode(prompt, return_tensors='pt')

# Réglage des paramètres pour diversifier les sorties tout en restant cohérent
temperature = 0.9  # Un bon point de départ pour l'équilibre entre variété et cohérence
top_k = 100  # Considère les 50 meilleurs mots à chaque étape pour plus de diversité
top_p = 0.92  # Utilise le nucleus sampling pour une génération plus dynamique
no_repeat_ngram_size = 2  # Permet une certaine répétition, mais pas immédiate

outputs = model.generate(
    inputs,
    max_length=85,
    num_return_sequences=5,  # Générer plusieurs séquences
    no_repeat_ngram_size=2,  # Permet une certaine répétition, mais pas immédiate
    temperature=0.9,  # Un bon point de départ pour l'équilibre entre variété et cohérence
    top_k=50,  # Considère les 50 meilleurs mots à chaque étape pour plus de diversité
    top_p=0.92,  # Utilise le nucleus sampling pour une génération plus dynamique
    do_sample=True,  # Activer l'échantillonnage pour permettre la génération de multiples séquences
)

for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated text {i+1}: {generated_text}")
