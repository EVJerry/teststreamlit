import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.title("Phân tích Cảm xúc Gương mặt AI")

uploaded_file = st.file_uploader("Tải ảnh gương mặt...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Chuyển đổi ảnh
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("Phân tích cảm xúc"):
        with st.spinner("Đang phân tích..."):
            try:
                # DeepFace phân tích cảm xúc
                results = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
                
                # Kết quả thường là một danh sách các khuôn mặt
                for face in results:
                    emotion = face['dominant_emotion']
                    # Chuyển đổi tên cảm xúc sang tiếng Việt cho thân thiện
                    emotion_dict = {
                        'angry': 'Giận dữ 😡', 'disgust': 'Ghê tởm 🤢', 
                        'fear': 'Sợ hãi 😨', 'happy': 'Hạnh phúc 😊', 
                        'sad': 'Buồn bã 😢', 'surprise': 'Bất ngờ 😲', 
                        'neutral': 'Bình thường 😐'
                    }
                    translated_emotion = emotion_dict.get(emotion, emotion)
                    
                    st.subheader(f"Cảm xúc chủ đạo: {translated_emotion}")
                    
                    # Hiển thị biểu đồ các chỉ số cảm xúc
                    st.bar_chart(face['emotion'])
            except Exception as e:
                st.error("Không tìm thấy gương mặt trong ảnh. Hãy thử ảnh khác!")
