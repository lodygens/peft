"""
Ce script n'a d'autre utilité que de montrer comment charger une config LoRA
"""

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftConfig
import json

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

with open('lora_config.json', 'r') as f:
    lora_config_dict = json.load(f)

print(lora_config_dict)
lora_config = LoraConfig(**lora_config_dict)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
