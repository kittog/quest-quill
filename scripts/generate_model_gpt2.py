# Importation du modèle entraîné/finetuné
model_path = "gpt2test"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generer_quete(map: str, difficulty: str, target: str) -> str:
	# Prompt de génération de la quête
    prompt = f"Map: {map} | Difficulty: {difficulty} | Target: {target} |"
    	# Encodage du prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    	# Génération correspondante avec paramètres de notre choix
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=10,
        temperature=0.9,
        top_k=10,
        top_p=0.92,
        do_sample=True,
    )

	# Décodage de la séquence générée
    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence
