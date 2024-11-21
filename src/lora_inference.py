"""
This script compares two models
1. facebook/opt-350m model.
2. and its LoRA version stored in ./opt-350m-lora by ./lora_train.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, prompt, max_length=50):
    """Génère une réponse à partir d'un modèle donné."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Charger le modèle de base
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Charger le modèle avec LoRA (entraîné précédemment)
    lora_model_path = "./opt-350m-lora"  # Chemin où le modèle LoRA a été sauvegardé
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

    # Demander un prompt à l'utilisateur
    prompt = input("Entrez un prompt : ")

    print("\n--- Réponse avec le modèle de base ---")
    base_response = generate_response(base_model, tokenizer, prompt)
    print(base_response)

    print("\n--- Réponse avec le modèle amélioré (LoRA) ---")
    lora_response = generate_response(lora_model, tokenizer, prompt)
    print(lora_response)

if __name__ == "__main__":
    main()
