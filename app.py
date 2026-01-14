import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Cấu hình trang rộng và tiêu đề chuyên nghiệp
st.set_page_config(page_title="Hệ thống Nhận diện Vật thể AI", layout="wide")

# Thiết kế Thanh bên (Sidebar)
with st.sidebar:
    st.title("⚙️ Cấu hình")
    st.info("Ứng dụng sử dụng mô hình YOLOv8 để nhận diện vật thể thời gian thực.")
    confidence = st.slider("Confidence", 0.0, 1.0, 0.4)
    st.divider()
    st.success("Trạng thái: Đang hoạt động")

# Tiêu đề chính
st.title("🔍 AI Vision Dashboard")
st.markdown("---")

# Chia cột cho ảnh gốc và kết quả
col1, col2 = st.columns(2)

uploaded_file = st.file_uploader("Tải ảnh lên để phân tích...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("🖼️ Ảnh đầu vào")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Kết quả AI")
        with st.spinner("Đang phân tích..."):
            # Load model
            model = YOLO("yolov8n.pt")
            results = model(image, conf=confidence)
            
            # Vẽ kết quả
            res_plotted = results[0].plot()
            st.image(res_plotted, channels="BGR", use_container_width=True)
            
            # Hiển thị thống kê
            count = len(results[0].boxes)
            st.metric("Số vật thể phát hiện được", count)
