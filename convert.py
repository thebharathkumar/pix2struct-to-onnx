import torch
from transformers import Pix2StructForDocVQA
import onnx
from transformers.convert_graph_to_onnx import convert

def convert_to_onnx(model_name, output_folder):
    # Load the Hugging Face model
    model = Pix2StructForDocVQA.from_pretrained(model_name)
    
    # Dummy input tensors (modify according to model's input requirements)
    input_shapes = {"input_ids": (1, 128), "attention_mask": (1, 128)}
    inputs = {input_name: torch.randn(shape) for input_name, shape in input_shapes.items()}
    
    # Convert the model to ONNX format
    onnx_model_path = f"{output_folder}/model.onnx"
    convert(framework="pt", model=model, output=onnx_model_path, opset=11, input_names=list(input_shapes.keys()), dynamic_axes={input_name: {0: "batch_size"} for input_name in input_shapes.keys()})
    
    print(f"Model converted to ONNX format and saved at: {onnx_model_path}")

if __name__ == "__main__":
    model_name = "google/pix2struct-docvqa-base"
    output_folder = "model"
    convert_to_onnx(model_name, output_folder)
