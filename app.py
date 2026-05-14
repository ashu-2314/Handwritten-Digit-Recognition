import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = tf.keras.models.load_model("model/mnist_cnn.h5")

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

/* REMOVE EXTRA TOP SPACE */

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}

/* Reduce gap above title */

.main-title {
    margin-top: 0px !important;
    margin-bottom: 5px !important;
}

/* Reduce subtitle spacing */

.subtitle {
    margin-top: 0px !important;
    margin-bottom: 20px !important;
}

/* Reduce badge spacing */

.badge {
    margin-top: 0px !important;
    margin-bottom: 15px !important;
}

/* ---------------- MAIN APP ---------------- */

.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
    color: #1e293b;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #1e1b4b, #312e81);
    color: white;
}

/* ---------------- MAIN TITLE ---------------- */

.main-title {
    text-align: center;
    font-size: 72px;
    font-weight: 800;
    margin-top: 10px;

    background: linear-gradient(
        to right,
        #ec4899,
        #7c3aed,
        #2563eb
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 10px;
}

/* ---------------- SUBTITLE ---------------- */

.subtitle {
    text-align: center;
    font-size: 24px;
    color: #475569;
    margin-bottom: 40px;
}

/* ---------------- BADGE ---------------- */

.badge {
    width: fit-content;
    margin: auto;
    padding: 10px 22px;
    border-radius: 30px;

    background: rgba(124, 58, 237, 0.08);

    color: #7c3aed;
    font-weight: 600;

    margin-top: 10px;
    margin-bottom: 20px;

    border: 1px solid rgba(124, 58, 237, 0.15);
}

/* ---------------- CARDS ---------------- */

.card {

    background: rgba(255,255,255,0.75);

    border-radius: 25px;

    padding: 35px;

    backdrop-filter: blur(10px);

    box-shadow: 0px 10px 30px rgba(0,0,0,0.06);

    border: 1px solid rgba(255,255,255,0.5);

    margin-bottom: 25px;
}

/* ---------------- UPLOAD TITLE ---------------- */

.upload-title {

    font-size: 30px;

    font-weight: 700;

    margin-bottom: 12px;

    color: #312e81;
}

/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {

    background: white;

    border-radius: 20px;

    padding: 15px;

    border: 2px dashed #c4b5fd;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* ---------------- BROWSE BUTTON ---------------- */

[data-testid="stFileUploader"] button {

    background: linear-gradient(
        to right,
        #7c3aed,
        #2563eb
    ) !important;

    color: white !important;

    border-radius: 12px !important;

    border: none !important;

    padding: 10px 18px !important;

    font-weight: 600 !important;

    transition: 0.3s ease;
}

/* Hover Effect */

[data-testid="stFileUploader"] button:hover {

    transform: scale(1.05);

    opacity: 0.95;
}

/* ---------------- PREDICTION BOX ---------------- */

.prediction-box {

    background: white;

    border-radius: 30px;

    padding: 35px;

    text-align: center;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
}

/* ---------------- DIGIT CIRCLE ---------------- */

.circle {

    width: 180px;

    height: 180px;

    margin: auto;

    border-radius: 50%;

    background: linear-gradient(
        135deg,
        #ec4899,
        #7c3aed
    );

    display: flex;

    justify-content: center;

    align-items: center;

    color: white;

    font-size: 82px;

    font-weight: bold;

    box-shadow: 0px 10px 25px rgba(124,58,237,0.25);
}

/* ---------------- CONFIDENCE ---------------- */

.confidence {

    margin-top: 22px;

    font-size: 26px;

    font-weight: 700;

    color: #7c3aed;
}

/* ---------------- FEATURE CARDS ---------------- */

.feature-card {

    background: white;

    padding: 28px;

    border-radius: 22px;

    box-shadow: 0px 8px 25px rgba(0,0,0,0.05);

    transition: 0.3s ease;

    height: 190px;

    border: 1px solid #f1f5f9;
}

/* Hover */

.feature-card:hover {

    transform: translateY(-8px);

    box-shadow: 0px 15px 35px rgba(0,0,0,0.08);
}

/* ---------------- FOOTER ---------------- */

.footer {

    position: fixed;

    bottom: 0;

    left: 0;

    width: 100%;

    text-align: center;

    padding: 14px;

    color: #64748b;

    font-size: 15px;

    background: rgba(255,255,255,0.8);

    backdrop-filter: blur(10px);

    border-top: 1px solid #e2e8f0;
}

/* ---------------- REMOVE STREAMLIT BRANDING ---------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.markdown("## 🧠 AI Digit Recognition")

    st.markdown("---")

    st.markdown("## ℹ️ About Project")

    st.write("""
This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from images.
""")

    st.markdown("---")

    st.markdown("## ⚡ Technologies")

    st.write("""
- Python
- TensorFlow
- CNN
- Streamlit
- NumPy
- Pillow
""")

    st.markdown("---")

    st.markdown("## 🎯 Accuracy")

    st.progress(99)

    st.write("Model Accuracy: 98.8%")

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="badge">✨ Powered by Deep Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image of a digit (0-9) and let our AI model predict it instantly.</div>',
    unsafe_allow_html=True
)

# ---------------- MAIN CONTENT ---------------- #

left, right = st.columns([2,1])

# ---------- LEFT ---------- #

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="upload-title">📤 Upload Digit Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Supported formats: PNG, JPG, JPEG",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert('L')

# Convert image to numpy array
        image_array = np.array(image)

# Apply threshold
        image_array = np.where(image_array > 150, 255, 0)

# Find digit coordinates
        coords = np.argwhere(image_array == 0)

# Crop digit
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        cropped = image_array[y0:y1, x0:x1]

# Convert back to PIL image
        cropped_image = Image.fromarray(cropped.astype(np.uint8))

# Resize digit
        cropped_image = cropped_image.resize((20, 20))

# Create centered 28x28 image
        new_image = Image.new('L', (28, 28), 255)

# Paste centered digit
        new_image.paste(cropped_image, (4, 4))

        st.markdown("### 🖼 Processed Image")

        st.image(new_image, width=220)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RIGHT ---------- #

with right:

    if uploaded_file is not None:

        image_array = np.array(new_image)

# Invert colors for MNIST
        image_array = 255 - image_array

# Normalize
        image_array = image_array / 255.0

# Reshape
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = model.predict(image_array)

        predicted_digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.markdown("""
        <div class="prediction-box">
        <h2 style="color:#7b2cbf;">Prediction Result</h2>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="circle">
            {predicted_digit}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="confidence">
            Confidence Score<br>
            {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE SECTION ---------------- #

st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown("""
    <div class="feature-card">
    <h2>⚡ High Accuracy</h2>
    <p>Achieves nearly 99% accuracy on MNIST dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
    <h2>🚀 Fast Prediction</h2>
    <p>Instant predictions using optimized CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
    <h2>🛡 Reliable Model</h2>
    <p>Trained on 60,000+ handwritten digit images.</p>
    </div>
    """, unsafe_allow_html=True)

with f4:
    st.markdown("""
    <div class="feature-card">
    <h2>🧠 Deep Learning</h2>
    <p>Built using TensorFlow and CNN architecture.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">
© 2026 AI Digit Recognition • Built with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)