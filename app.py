# app.py
import streamlit as st
from PIL import Image
import os
from style_transfer import run_style_transfer
from upscaler import upscale_image

# Paths
STYLE_IMAGE = "style_images/style.jpg"
os.makedirs("output_images", exist_ok=True)

st.set_page_config(page_title="AI Image Stylizer & Upscaler", layout="centered")

st.title("🎨 AI Image Stylizer + 🔼 Super-Resolution")
st.markdown("Upload an image, apply artistic style, and upscale it using Swin2SR (Hugging Face).")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

stylized_img = None

if uploaded_file is not None:
    content_image = Image.open(uploaded_file)
    st.image(content_image, caption="Original Image", use_column_width=True)

    if st.button("✨ Apply Artistic Style"):
        with st.spinner("Applying style transfer..."):
            stylized_img = run_style_transfer(content_image, STYLE_IMAGE, steps=150)
            stylized_img.save("output_images/stylized_output.jpg")
        st.success("Style applied successfully!")
        st.image(stylized_img, caption="Stylized Image", use_column_width=True)
        st.download_button("📥 Download Stylized Image", stylized_img, file_name="stylized_output.jpg")

if os.path.exists("output_images/stylized_output.jpg"):
    if st.button("🔼 Upscale Stylized Image"):
        with st.spinner("Upscaling with Swin2SR..."):
            output_image = upscale_image(Image.open("output_images/stylized_output.jpg"))
            output_image.save("output_images/upscaled_output.jpg")
        st.success("Upscaled image ready!")
        st.image(output_image, caption="Upscaled Image", use_column_width=True)
        st.download_button("📥 Download Upscaled Image", output_image, file_name="upscaled_output.jpg")
