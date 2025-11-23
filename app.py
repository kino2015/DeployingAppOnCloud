import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
import os

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# ----------------- モデルのロード -----------------
MODEL_PATH = "model.onnx" 

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# ----------------- 認識処理関数 -----------------

def recognize_digit(image_np):
    """
    Flaskと同じシンプルな前処理
    """
    if image_np is None:
        return "No image uploaded", 0.0
    
    # 1. グレースケールに変換
    if len(image_np.shape) == 3:
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        img = image_np
    
    # 2. blobFromImageと同じ処理（28x28にリサイズして正規化）
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    
    # 3. 入力テンソルの整形 (1, 1, 28, 28)
    blob = normalized.reshape(1, 1, 28, 28).astype(np.float32)
    
    # 4. ONNXモデルによる予測
    out = session.run([output_name], {input_name: blob})[0]
    
    # 5. softmax を適用
    probabilities = softmax(out.flatten())
    
    # 6. 結果の解釈
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit]
    
    # 7. Top-5の予測結果
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_results = "\n".join([
        f"Digit {idx}: {probabilities[idx]:.4f}" 
        for idx in top5_indices
    ])
    
    result_text = f"Recognized Digit: {predicted_digit}\n\nTop 5 Predictions:\n{top5_results}"
    
    return result_text, confidence

# ----------------- Gradioインターフェース -----------------

interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(type="numpy", label="Upload Handwritten Digit (Grayscale)"),
    outputs=[
        gr.Textbox(label="Recognition Result", lines=8),
        gr.Number(label="Confidence")
    ],
    title="🔢 Digit Recognizer (Huggingface Spaces / Gradio)",
    description="Upload an image of a handwritten digit to get the prediction."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
