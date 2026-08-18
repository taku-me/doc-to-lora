#!/usr/bin/env python3
"""Process PDF text with gemma D2L model"""
import torch
import sys
import os

sys.path.insert(0, '/Volumes/NVME202502/projects/doc-to-lora/src')

from ctx_to_lora.model_loading import get_tokenizer, get_model
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# Load model
checkpoint_path = "/Volumes/NVME202502/projects/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
print("Loading model...")
state_dict = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

# Load base model without flash attention
base_model = get_model(
    state_dict,
    train=False,
    requires_grad=False,
    use_flash_attn=False,  # Disable flash attention
)
print(f"Base model: {base_model.config._name_or_path}")

# Create D2L model
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
# Replace base model with the one loaded without flash attn
model.base_model = base_model
model.reset()

tokenizer = get_tokenizer(model.base_model.name_or_path)
print(f"Tokenizer: {tokenizer}")
print(f"Device: {model.device}")

# Load extracted text
text_path = "/Volumes/NVME202502/projects/doc-to-lora/tmp_xtalk_text.txt"
doc = open(text_path, 'r', encoding='utf-8').read()
print(f"Loaded text: {len(doc)} characters")

# Internalize the document
print("Internalizing document...")
model.internalize(doc)
print("Done internalizing")

# Test queries
queries = [
    "この文書の内容を要約してください。",
    "xTalkとは何ですか？",
    "本多さんが話している内容は何ですか？",
    "ろう・難聴者のプロジェクトについて教えてください。",
]

for q in queries:
    chat = [{"role": "user", "content": q}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    
    print(f"\n=== Query: {q} ===")
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
    response = tokenizer.decode(outputs[0])
    print(response)
    print("-" * 50)
