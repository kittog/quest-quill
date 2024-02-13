from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model_name = "gpt2-medium"  # Utilisez "gpt2-large" pour le modèle large
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

train_path = 'test.txt'  # Mettez ici le chemin vers vos données reformulées

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2_quest_generator_finetuned2",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=10,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Sauvegarder le modèle finetuné et le tokenizer
model.save_pretrained("./gpt2_quest_generator_finetuned2")
tokenizer.save_pretrained("./gpt2_quest_generator_finetuned2")
