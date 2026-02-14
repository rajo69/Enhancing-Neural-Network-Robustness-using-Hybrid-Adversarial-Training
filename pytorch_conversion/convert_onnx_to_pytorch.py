import os
import glob
import torch
import onnx
import numpy as np
from onnx2pytorch import ConvertModel

print("\n=== MATLAB ONNX → PyTorch Converter ===\n")

# ===== CONFIGURATION =====
MODEL_DIR = "."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 ResNet input size
INPUT_SHAPE = (1, 3, 32, 32)


def load_and_validate_onnx(path):
    """
    Load ONNX and verify integrity
    """
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model
    except Exception as e:
        raise RuntimeError(f"Invalid ONNX file: {e}")


def convert_model(onnx_path):
    """
    Convert ONNX to PyTorch safely
    """
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    print(f"\nProcessing model: {model_name}")

    # Load ONNX
    onnx_model = load_and_validate_onnx(onnx_path)
    print("✓ ONNX validation passed")

    # Convert to PyTorch
    pytorch_model = ConvertModel(onnx_model)
    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    print("✓ Converted to PyTorch")

    # Save weights
    output_file = f"{model_name}.pth"
    torch.save(pytorch_model.state_dict(), output_file)

    print(f"✓ Saved PyTorch weights → {output_file}")

    return pytorch_model, model_name


def verify_inference(model, model_name):
    """
    Run dummy inference to ensure model correctness
    """
    try:
        dummy_input = torch.randn(INPUT_SHAPE).to(DEVICE)

        with torch.no_grad():
            output = model(dummy_input)

        print("✓ Inference test passed")
        print(f"Output shape: {tuple(output.shape)}")

    except Exception as e:
        print(f"⚠ Inference failed for {model_name}: {e}")


def main():
    # Find ONNX files
    onnx_files = glob.glob(os.path.join(MODEL_DIR, "*.onnx"))

    if not onnx_files:
        print("❌ No ONNX models found in root folder.")
        return

    print(f"Found {len(onnx_files)} ONNX model(s).")

    success_count = 0

    for path in onnx_files:
        try:
            model, name = convert_model(path)
            verify_inference(model, name)
            success_count += 1

        except Exception as e:
            print(f"\n❌ Failed to convert {path}")
            print("Error:", e)

    print("\n===================================")
    print(f"Successfully converted {success_count}/{len(onnx_files)} models")
    print("===================================")


if __name__ == "__main__":
    main()
