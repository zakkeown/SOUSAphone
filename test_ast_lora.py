#!/usr/bin/env python3
"""Quick test to validate AST + LoRA configuration works."""
import torch
from transformers import ASTForAudioClassification
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("Testing AST + LoRA Configuration")
print("=" * 80)

# Load AST model
print("\n1. Loading AST model...")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=40,
    ignore_mismatched_sizes=True,
)
print(f"   ✓ Model loaded: {model.__class__.__name__}")

# Count original parameters
original_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params: {original_params:,}")
print(f"   Trainable params (before LoRA): {trainable_params:,}")

# Create LoRA config with correct target modules
print("\n2. Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],  # Will match across all layers
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
print(f"   LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
print(f"   Target modules: {lora_config.target_modules}")

# Apply PEFT
try:
    model = get_peft_model(model, lora_config)
    print("   ✓ LoRA applied successfully!")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    exit(1)

# Count new parameters
lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
reduction = 100 * (1 - lora_params / trainable_params)
print(f"\n3. Parameter Report:")
print(f"   Total params: {original_params:,}")
print(f"   Trainable params (with LoRA): {lora_params:,}")
print(f"   Reduction: {reduction:.2f}% fewer trainable params")
print(f"   Efficiency: {100 * lora_params / original_params:.3f}% of total params trainable")

# Test forward pass
print("\n4. Testing forward pass...")
batch_size = 2
time_frames = 1024  # AST expects 1024 time frames (from config: max_length: 1024)
n_mels = 128

dummy_input = torch.randn(batch_size, time_frames, n_mels)
print(f"   Input shape: {tuple(dummy_input.shape)}")

try:
    with torch.no_grad():
        # AST expects input_values kwarg
        outputs = model(input_values=dummy_input)
        logits = outputs.logits
    print(f"   Output shape: {tuple(logits.shape)}")
    print(f"   ✓ Forward pass successful!")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Print some LoRA modules
print("\n5. LoRA Modules Added:")
lora_modules = [name for name, _ in model.named_modules() if 'lora' in name.lower()]
print(f"   Found {len(lora_modules)} LoRA modules")
for module in lora_modules[:5]:
    print(f"   - {module}")
if len(lora_modules) > 5:
    print(f"   ... and {len(lora_modules) - 5} more")

print("\n" + "=" * 80)
print("✅ SUCCESS: AST + LoRA configuration is working!")
print("=" * 80)
print("\nNext steps:")
print("1. Run a quick 2-epoch training test with: python3 train.py model=ast_fixed data=tiny training.max_epochs=2")
print("2. Check metrics are improving")
print("3. If successful, expand to other models")
