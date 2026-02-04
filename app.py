import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI Emotion Selector", layout="wide")

# --- SIDEBAR: Nơi chọn cảm xúc ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    st.write("Chọn các cảm xúc bạn muốn hệ thống hiển thị:")
    
    # Tạo danh sách các checkbox
    check_angry = st.checkbox("Giận dữ 😡", value=True)
    check_fear = st.checkbox("Sợ hãi 😨", value=True)
    check_happy = st.checkbox("Vui vẻ 😊", value=True)
    check_sad = st.checkbox("Buồn bã 😢", value=True)
    check_surprise = st.checkbox("Bất ngờ 😲", value=True)
    check_neutral = st.checkbox("Bình thường 😐", value=True)

    # Gom các lựa chọn vào một danh sách để đối chiếu
    selected_emotions = []
    if check_angry: selected_emotions.append('angry')
    if check_fear: selected_emotions.append('fear')
    if check_happy: selected_emotions.append('happy')
    if check_sad: selected_emotions.append('sad')
    if check_surprise: selected_emotions.append('surprise')
    if check_neutral: selected_emotions.append('neutral')

# --- GIAO DIỆN CHÍNH ---
st.title("🔍 Nhận diện Cảm xúc Gương mặt (Tùy chọn)")

uploaded_file = st.file_uploader("Tải ảnh gương mặt lên...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Ảnh gốc")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 Kết quả phân tích")
        if st.button("Bắt đầu phân tích"):
            with st.spinner("Đang xử lý..."):
                try:
                    results = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
                    
                    for face in results:
                        dominant = face['dominant_emotion']
                        
                        # Kiểm tra xem cảm xúc chủ đạo có nằm trong danh sách được chọn không
                        if dominant in selected_emotions:
                            st.success(f"Cảm xúc phát hiện: **{dominant.upper()}**")
                        else:
                            st.warning(f"Cảm xúc chủ đạo là '{dominant}', nhưng bạn đã bỏ chọn hiển thị cảm xúc này.")
                        
                        # Chỉ hiển thị biểu đồ của các cảm xúc đã chọn
                        full_stats = face['emotion']
                        filtered_stats = {k: v for k, v in full_stats.items() if k in selected_emotions}
                        
                        if filtered_stats:
                            st.write("Chỉ số chi tiết các cảm xúc đã chọn:")
                            st.bar_chart(filtered_stats)
                        else:
                            st.info("Hãy chọn ít nhất một cảm xúc ở thanh bên để xem biểu đồ chi tiết.")
                            
                except Exception as e:
                    st.error("Không thể nhận diện gương mặt. Vui lòng thử ảnh rõ hơn!")
