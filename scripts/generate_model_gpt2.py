
model_path = "gpt2test"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generer_quete(map: str, difficulty: str, target: str) -> str:
    prompt = f"Map: {map} | Difficulty: {difficulty} | Target: {target} |"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
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

    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence
