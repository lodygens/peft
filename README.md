# PEFT 

A project to understand and test large pretrained model adaptation with 
PEFT (Parameter-Efficient Fine-Tuning) LoRA

## Reference
[Higging Face Parameter-Efficient Fine-Tuning page](https://huggingface.co/docs/peft/main/en/index)

## Config
```
 python3.12 -m venv .
 source bin/activate
 pip install --upgrade pip
 pip install -r requirements.txt
```

##  Specialize facebook/opt-350m model

```
python src/lora_train.py
```

##  Compare facebook/opt-350m model and its specialized version
```
(peft) √ peft % python src/lora_inference.py
Entrez un prompt : what is the essence of life

--- Réponse avec le modèle de base ---
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
what is the essence of life ? ; the spirit of life ; the soul of life ; and the soul of death . 
assing 
assed 

--- Réponse avec le modèle amélioré (LoRA) ---
what is the essence of life ? ; that is , the essence of life itself . These words come from the 19th century philosopher Charles Simons , who coined the term “ love ” . In his book “ Love ” ,
````
