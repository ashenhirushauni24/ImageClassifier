
import cv2
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

st.set_page_config(
    page_title="Visionyx | AI Image Classifier",
    page_icon="◈",
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background: #0B1019;
        color: #F1F5F9;
    }

    .block-container {
        max-width: 760px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    h1, h2, h3 {
        letter-spacing: -0.04em;
    }

    .brand {
        font-size: 1.15rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        color: #F1F5F9;
        margin-bottom: 4rem;
    }

    .brand span {
        color: #60A5FA;
    }

    .hero-label {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid #27364B;
        border-radius: 100px;
        color: #93C5FD;
        background: #152236;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: clamp(2.4rem, 7vw, 4rem);
        font-weight: 750;
        letter-spacing: -0.065em;
        line-height: 1.12;
        margin-bottom: 1rem;
        color: #F8FAFC;
    }

    .hero-title span {
        color: #60A5FA;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        line-height: 1.7;
        margin-bottom: 3rem;
    }

    [data-testid="stFileUploader"] {
        background: #121B2A;
        border: 1px solid #263449;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #121B2A;
        border: 1px dashed #40516B;
        border-radius: 12px;
        padding: 24px 12px;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #25344B;
        color: #E2E8F0;
        border: 1px solid #40516B;
        border-radius: 8px;
    }

    div.stButton > button {
        width: 100%;
        background: #3B82F6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 650;
        transition: background 0.2s ease, transform 0.2s ease;
    }

    div.stButton > button:hover {
        background: #2563EB;
        color: white;
        transform: translateY(-1px);
    }

    div.stButton > button:focus {
        color: white;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 650;
        color: #F1F5F9;
        margin: 2rem 0 1rem;
        letter-spacing: -0.03em;
    }

    .prediction-label {
        color: #CBD5E1;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .footer-note {
        text-align: center;
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 4rem;
    }

    hr {
        border-color: #253247;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)


def classify_image(model, image):
    try:
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)
        return decode_predictions(predictions, top=3)[0]

    except Exception as e:
        st.error(f"Classification failed: {e}")
        return None


def main():
    st.markdown(
        '<div class="brand">◈ VISION<span>YX</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="hero-label">✦ AI-POWERED IMAGE RECOGNITION</div>
        <div class="hero-title">
            See beyond<br>
            the <span>pixels.</span>
        </div>
        <div class="hero-subtitle">
            Upload any image and discover what artificial
            intelligence sees. Simple, fast, and intelligent.
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"],
        max_upload_size=10
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="section-heading">Image preview</div>',
            unsafe_allow_html=True
        )

        st.image(image, use_container_width=True)

        if st.button("✦ Analyze Image", type="primary"):
            with st.spinner("Analyzing your image..."):
                try:
                    model = load_model()
                    predictions = classify_image(model, image)
                except Exception as e:
                    st.error(f"Unable to load model: {e}")
                    predictions = None

            if predictions:
                st.markdown(
                    '<div class="section-heading">Analysis results</div>',
                    unsafe_allow_html=True
                )

                st.caption("Top 3 predicted ImageNet categories")

                for index, (_, label, score) in enumerate(predictions):
                    label = label.replace("_", " ").title()

                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(
                            f"**{label}**"
                        )

                    with col2:
                        st.markdown(f"**{score:.1%}**")

                    st.progress(float(score))

                st.success("Analysis completed successfully.")

    st.markdown("""
        <div class="footer-note">
            Powered by MobileNetV2 · TensorFlow · Streamlit<br>
            See beyond the pixels.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
