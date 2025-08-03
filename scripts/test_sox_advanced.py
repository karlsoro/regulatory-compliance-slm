#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_sox_model():
    print("Loading SOX model...")
    model_dir = 'models/checkpoints'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    print("Model loaded successfully!")
    
    # Test prompts
    prompts = [
        "SOX compliance requirements for financial reporting include:",
        "Internal controls for ERP systems must:",
        "Documentation requirements for SOX compliance:"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(
            **inputs, 
            max_length=120, 
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

if __name__ == "__main__":
    test_sox_model()
