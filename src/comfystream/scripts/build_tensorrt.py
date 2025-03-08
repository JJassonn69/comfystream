import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
import sys
sys.path.append('../..')
from distillanydepth.modeling.archs.dam.dam import DepthAnything

def convert_safetensors_to_onnx(safetensors_path, onnx_path, encoder_type, input_size=(518, 518)):
    """
    Converts a DepthAnything model from safetensors format to ONNX format.
    """

    device = 'cpu'  # Export to ONNX on CPU is generally recommended

    # 1. Instantiate the DepthAnything model architecture
    try:
        model = DepthAnything(encoder=encoder_type).to(device)
    except Exception as e:
        print(f"Error instantiating DepthAnything model with encoder '{encoder_type}': {e}")
        print("Please ensure the encoder type is correct and the model code is correctly defined.")
        return

    # 2. Load weights from the safetensors file
    try:
        state_dict = safetensors.torch.load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {safetensors_path}")
    except Exception as e:
        print(f"Error loading weights from safetensors file '{safetensors_path}': {e}")
        print("Please check if the safetensors file path is correct and the file is valid.")
        return

    model.eval()  # Set the model to evaluation mode

    # 3. Create a dummy input tensor
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # 4. Export the model to ONNX format
    try:
        torch.onnx.export(
            model,                      # model being run
            (dummy_input,),              # model input (tuple for multiple inputs)
            onnx_path,                  # where to save the model
            export_params=True,         # store the trained parameter weights inside the ONNX file
            opset_version=11,           # the ONNX version to export to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=['input'],  # input node names
            output_names=['output'], # output node names
            # dynamic_axes={'input_image': {0: 'batch', 2: 'height', 3: 'width'},    # dynamic axes - REMOVE THIS
            #               'depth_output': {0: 'batch', 2: 'height', 3: 'width'}} - REMOVE THIS
        )
        print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Please check if your PyTorch environment and ONNX export settings are correct.")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DepthAnything safetensors to ONNX.")
    parser.add_argument("--safetensors_path", type=str, required=True, help="Path to the input safetensors file.")
    parser.add_argument("--onnx_path", type=str, default="depth_anything.onnx", help="Path to save the output ONNX file. Defaults to depth_anything.onnx.")
    parser.add_argument("--encoder_type", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"], help="Type of ViT encoder (vits, vitb, vitl, vitg). Defaults to vitl.")
    parser.add_argument("--input_height", type=int, default=518, help="Input image height. Defaults to 518.") # Fixed Height
    parser.add_argument("--input_width", type=int, default=518, help="Input image width. Defaults to 518.")  # Fixed Width

    args = parser.parse_args()

    convert_safetensors_to_onnx(
        args.safetensors_path,
        args.onnx_path,
        args.encoder_type,
        input_size=(args.input_height, args.input_width)
    )