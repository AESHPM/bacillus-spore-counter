
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Bacillus Endospore Counter", layout="centered")
st.title("🔬 Bacillus Endospore Counter")
st.markdown("Upload a microscopy image to detect and count Bacillus endospores.")

@st.cache_resource
def load_model():
    return YOLO("endospore-detector.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def process_image(image, model):
    img = Image.open(image).convert("RGB")
    results = model.predict(img, conf=0.25, verbose=False)[0]
    count = len(results.boxes) if results.boxes is not None else 0
    img_array = np.array(img)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        label = f"Spore: {conf:.2f}"
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return count, img_array

if uploaded_file:
    st.image(uploaded_file, caption="Input Image", use_column_width=True)
    with st.spinner("Detecting endospores..."):
        count, processed = process_image(uploaded_file, model)
        st.success(f"Detected endospores: {count}")
        st.image(processed, caption="Detection Results", use_column_width=True)
