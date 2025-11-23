# app.py

import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
import os

# ----------------- モデルのロード -----------------
# ONNXモデルのパスを定義
MODEL_PATH = "model.onnx" 

# モデルが存在するか確認（Huggingface Spacesではパスが重要）
if not os.path.exists(MODEL_PATH):
    # エラー処理。通常は不要だが、デバッグ用
    print(f"Error: Model file not found at {MODEL_PATH}")

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    # ロード失敗時はプログラムを終了させず、デバッグ情報を出す
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# ----------------- 認識処理関数 -----------------

def recognize_digit(image_np):
    """
    Gradioから受け取ったNumPy配列の画像を処理し、認識結果を返す関数。
    """
    if image_np is None:
        return "No image uploaded", 0.0
    
    # 1. グレースケールに変換
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np

    # 2. リサイズと前処理 (28x28、反転、正規化)
    resized_image = cv2.resize(image_gray, (28, 28), interpolation=cv2.INTER_AREA)
    preprocessed_image = 1 - (resized_image / 255.0) 
    
    # 3. 入力テンソルの整形 (1, 1, 28, 28)
    input_tensor = preprocessed_image.reshape(1, 1, 28, 28).astype(np.float32)

    # 4. ONNXモデルによる予測
    input_dict = {input_name: input_tensor}
    raw_output = session.run([output_name], input_dict)[0]
    
    # 5. 結果の解釈
    predicted_digit = np.argmax(raw_output)
    confidence = np.max(raw_output)
    
    return f"Recognized Digit: {predicted_digit}", confidence

# ----------------- Gradioインターフェースの構築 -----------------

# インターフェースの定義: 入力は画像、出力は文字列と数値
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(type="numpy", label="Upload Handwritten Digit (Grayscale)"),
    outputs=[
        gr.Textbox(label="Recognition Result"),
        gr.Number(label="Confidence")
    ],
    title="🔢 Digit Recognizer (Huggingface Spaces / Gradio)",
    description="Upload an image of a handwritten digit to get the prediction."
)

# Gradioアプリの起動
if __name__ == "__main__":
    interface.launch()