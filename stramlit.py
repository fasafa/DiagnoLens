import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="🩺 AI Diagnosis", layout="centered")

st.title("🧠 AI Vision Diagnosis (Llama 3.2 + LoRA)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Diagnose"):
        with st.spinner("Analyzing image..."):
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                files={"image": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Diagnosis Complete")
                st.write("**Result:**", result.get("diagnosis", "No result"))
            else:
                st.error("❌ Backend error.")
