from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

try:
    model = GPT2LMHeadModel.from_pretrained("./gpt2-quests-saved")
except:
    model = GPT2LMHeadModel.from_pretrained("gpt2")

def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return dataset, data_collator

dataset, data_collator = load_dataset('quests.json', tokenizer)

training_args = TrainingArguments(
    output_dir="./gpt2-quests",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./gpt2-quests-saved")

# Génération de texte
input_ids = tokenizer.encode('Hunt', return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
