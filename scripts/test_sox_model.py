#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the trained model and tokenizer
model_dir = 'models/checkpoints'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Test prompt
prompt = "Generate SOX-compliant internal control documentation for ERP system"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)

# Decode and print the result
print("\n=== Model Output ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
