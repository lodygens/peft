"""
This script trains facebook/opt-350m model.
Dataset is loaded from Hugging Face
On an Apple M2, with 16Gb RAM, it took 37 minutes to train

LoRA result is stored in ./opt-350m-lora
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset

# Chargement du modèle et du tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Préparer le modèle pour l'entraînement (si nécessaire pour une quantification 8-bit)
model = prepare_model_for_kbit_training(model)

# Configuration de LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Application de LoRA au modèle
model = get_peft_model(model, lora_config)
print("Modèle PEFT appliqué avec LoRA !")

# Chargement d'un jeu de données
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")  # Exemple avec 5% des données
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:5%]")  # Jeu d'évaluation

# Prétraitement des données : Ajout du champ `labels`
def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()  # Les labels sont les mêmes que les input_ids
    return inputs

tokenized_data = dataset.map(preprocess_function, batched=True)
tokenized_eval_data = eval_dataset.map(preprocess_function, batched=True)


# Définition des paramètres d'entraînement
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Utilisation du paramètre mis à jour
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
)

# Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_eval_data,
    tokenizer=tokenizer,  # Gardé pour rétrocompatibilité
)

trainer.train()

# Sauvegarde du modèle ajusté
model.save_pretrained("./opt-350m-lora")

print("Entraînement terminé avec LoRA !")
