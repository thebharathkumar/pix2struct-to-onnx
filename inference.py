import argparse
import time
import numpy as np
import onnxruntime

def load_and_infer(onnx_model_path, image_path):
    # Load the ONNX model
    session = onnxruntime.InferenceSession(onnx_model_path)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_input = np.array(image)
    image_input = np.transpose(image_input, (2, 0, 1))
    image_input = np.expand_dims(image_input, axis=0).astype(np.float32)
    
    # Perform inference
    start_time = time.time()
    outputs = session.run(None, {"input_ids": image_input})
    end_time = time.time()
    
    inference_time = end_time - start_time
    return outputs, inference_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder", type=str, help="Path to ONNX model folder")
    parser.add_argument("-i", "--image_path", type=str, help="Path to input image")
    args = parser.parse_args()
    
    onnx_model_path = f"{args.model_folder}/model.onnx"
    outputs, inference_time = load_and_infer(onnx_model_path, args.image_path)
    
    print("Inference result:", outputs)
    print(f"Inference time: {inference_time:.4f} seconds")
