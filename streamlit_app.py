import streamlit as st
import numpy as np
import cv2 # OpenCVのインポートは必須
import onnxruntime as ort # ONNXランタイムのインポートは必須
import os # model.onnxの存在確認用に追加

# ----------------- モデルと関数の定義 -----------------

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # 数値的な安定性のために x から最大値を引く
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# ONNXモデルのロード
try:
    # 実際には、このディレクトリ内の 'model.onnx' を読み込む必要があります
    # Streamlit Cloudでは、GitHubのルートにあるファイルを相対パスで読み込みます。
    # 念のため、モデルファイルが存在するか確認
    if not os.path.exists("model.onnx"):
        st.error("モデルファイル 'model.onnx' が見つかりません。")
        st.stop()
        
    session = ort.InferenceSession("model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop() # エラーが発生したらアプリの実行を停止

def process_image(img_data):
    # 画像データをOpenCV形式に変換
    np_img = np.frombuffer(img_data.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    
    # 画像を28x28にリサイズし、ONNXモデルの入力形式に整形
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # 反転（白黒）、正規化 (0-1)、形状変更 (1, 1, 28, 28)
    preprocessed_image = 1 - (resized_image / 255.0) 
    input_tensor = preprocessed_image.reshape(1, 1, 28, 28).astype(np.float32)

    return input_tensor, resized_image

# ----------------- Streamlit UI -----------------

st.title("🔢 Digit Recognizer (Streamlit)")
st.subheader("Upload an image of a handwritten digit (0-9)")

# ファイルアップローダー
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 処理の実行
    with st.spinner('Processing image and predicting...'):
        # 画像の前処理
        input_tensor, display_img = process_image(uploaded_file)
        
        # ONNXモデルによる予測
        input_dict = {input_name: input_tensor}
        raw_output = session.run([output_name], input_dict)[0]
        
        # --- 修正箇所 ---
        # 1. raw_output(Logits)にsoftmaxを適用し、確率に変換
        probabilities = softmax(raw_output.flatten())
        
        # 2. 予測結果の解釈
        predicted_digit = np.argmax(probabilities)
        confidence = np.max(probabilities) # 修正: softmax 後の値を使用
        # --------------
        
    # 結果の表示
    st.image(display_img, caption="Processed Image (28x28)", width=150)
    
    # 信頼度は0-1の範囲になる
    st.success(f"Recognized Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.4f}")

    # 詳細なロジスティック（オプション）
    if st.checkbox('Show Raw Predictions'):
        st.write(probabilities)