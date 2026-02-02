#!/usr/bin/env python3
"""Test all 4 models with LoRA to ensure configurations work."""
import torch
from peft import LoraConfig, get_peft_model

# Model imports
from sousa.models.ast import ASTModel
from sousa.models.efficientat import EfficientATModel
from sousa.models.htsat import HTSATModel
from sousa.models.beats import BEATsModel

# Target modules for each model (from configs)
MODEL_CONFIGS = {
    "AST": {
        "class": ASTModel,
        "target_modules": ["query", "key", "value", "dense"],
        "input_shape": (2, 1024, 128),  # (batch, time, mels)
        "input_type": "spectrogram",
    },
    "EfficientAT": {
        "class": EfficientATModel,
        "target_modules": ["attention_pool.0", "attention_pool.2", "classifier.1"],
        "input_shape": (2, 1000, 128),  # (batch, time, mels)
        "input_type": "spectrogram",
    },
    "HTS-AT": {
        "class": HTSATModel,
        "target_modules": ["query", "key", "value", "dense"],
        "input_shape": (2, 256, 128),  # (batch, time, mels) - HTS-AT has smaller max size
        "input_type": "spectrogram",
    },
    "BEATs": {
        "class": BEATsModel,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "dense"],
        "input_shape": (2, 80000),  # (batch, samples) - raw waveform at 16kHz for 5sec
        "input_type": "waveform",
    },
}

def test_model(model_name, config):
    """Test a single model with LoRA."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name}")
    print('='*80)

    try:
        # Load model
        print(f"1. Loading {model_name} model...")
        model = config["class"](num_classes=40, pretrained=False)
        original_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model loaded: {original_params:,} parameters")

        # Apply LoRA
        print(f"\n2. Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=config["target_modules"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        print(f"   Target modules: {config['target_modules']}")

        model = get_peft_model(model, lora_config)
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        reduction = 100 * (1 - lora_params / original_params)
        print(f"   ✓ LoRA applied successfully!")
        print(f"   Trainable params: {original_params:,} → {lora_params:,}")
        print(f"   Reduction: {reduction:.2f}%")

        # Test forward pass
        print(f"\n3. Testing forward pass...")
        dummy_input = torch.randn(config["input_shape"])
        print(f"   Input shape: {tuple(dummy_input.shape)} ({config['input_type']})")

        with torch.no_grad():
            # PEFT wraps models with a forward() that expects input_ids (HF convention)
            # Our audio models don't use input_ids, so we bypass PEFT's forward()
            # by calling base_model.model() directly (same as training module does)
            if hasattr(model, 'base_model'):
                outputs = model.base_model.model(dummy_input)
            else:
                outputs = model(dummy_input)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

        print(f"   Output shape: {tuple(logits.shape)}")
        print(f"   ✓ Forward pass successful!")

        print(f"\n✅ {model_name} PASSED all tests!")
        return True

    except Exception as e:
        print(f"\n❌ {model_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("Testing All Models with LoRA")
    print("="*80)
    print(f"\nTesting {len(MODEL_CONFIGS)} models: {', '.join(MODEL_CONFIGS.keys())}")

    results = {}
    for model_name, config in MODEL_CONFIGS.items():
        results[model_name] = test_model(model_name, config)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    print(f"\nResults: {passed}/{total} models passed")
    print()

    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {model_name:15} {status}")

    if passed == total:
        print("\n🎉 SUCCESS: All models work with LoRA!")
        print("\nNext steps:")
        print("1. Run full sweep: py311 scripts/run_sweep.py --data tiny --max-epochs=10")
        print("2. All 4 models should now complete successfully")
        return 0
    else:
        print(f"\n⚠️  {total - passed} model(s) failed - need fixes before sweep")
        return 1

if __name__ == "__main__":
    exit(main())
