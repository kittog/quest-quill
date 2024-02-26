from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import matplotlib.pyplot as plt

model_name = "gpt2-medium" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

train_path = 'test_clean.txt' 

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=75
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
print(len(train_dataset))
training_args = TrainingArguments(
    output_dir="./model_75",
    overwrite_output_dir=True,
    num_train_epochs=20,
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

# Entraînement du modèle
trainer.train()

# Sauvegarder le modèle finetuné et le tokenizer
model.save_pretrained("./model_75")
tokenizer.save_pretrained("./model_75")

