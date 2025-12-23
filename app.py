import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv8 Object Detection on Streamlit")

# 1. Load mô hình (Sẽ tự động tải file .pt về server ở lần chạy đầu tiên)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

# 2. Giao diện tải ảnh
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chuyển file tải lên thành ảnh PIL
    image = Image.open(uploaded_file)
    
    # Hiển thị ảnh gốc
    st.image(image, caption="Ảnh gốc", use_column_width=True)
    
    if st.button("Bắt đầu nhận diện"):
        # 3. Chạy mô hình
        results = model(image)
        
        # 4. Vẽ kết quả lên ảnh
        res_plotted = results[0].plot() # Trả về mảng numpy (BGR)
        
        # Hiển thị ảnh kết quả
        st.image(res_plotted, caption="Kết quả nhận diện", channels="BGR", use_column_width=True)
        
        # Hiển thị danh sách vật thể tìm thấy
        st.write("Các vật thể phát hiện được:")
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0])
            st.write(f"- **{label}**: {conf:.2f}")
