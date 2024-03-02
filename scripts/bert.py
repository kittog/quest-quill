from transformers import BertForMaskedLM, BertTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Chargement du tokenizer et du modèle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Prétraitement des données pour BERT
def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    return dataset, data_collator

dataset, data_collator = load_dataset('quests_train.txt', tokenizer)

training_args = TrainingArguments(
    output_dir="./bert-quests",
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

output_model_dir = "./saved_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
import torch
from transformers import BertForMaskedLM, BertTokenizer

def generer_quete(model, tokenizer, map, difficulty, target):
    prompt = f"Créer une quête sur la carte {map}, avec une difficulté {difficulty}, ciblant {target}. La quête est : [CLS]"
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    outputs = model.generate(inputs, max_length=128, num_return_sequences=1)
    generated_quete = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_quete


model_path = "bert"  
model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

map = input("Entrez la carte : ")
difficulty = input("Entrez la difficulté (en chiffres) : ")
target = input("Entrez la cible : ")

quete_generee = generer_quete(model, tokenizer, map, difficulty, target)
print("\nQuête Générée :\n", quete_generee)