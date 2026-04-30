import streamlit as st
import numpy as np
import os
from PIL import Image
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Vehicle Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark theme overrides */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(59,130,246,0.1) 50%, rgba(16,185,129,0.1) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(239,68,68,0.05) 0%, transparent 60%);
    animation: pulse-bg 4s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.7; }
}

.hero-badge {
    display: inline-block;
    background: rgba(239,68,68,0.2);
    border: 1px solid rgba(239,68,68,0.5);
    color: #f87171;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f87171, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto 1.5rem;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(239,68,68,0.4);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #f87171;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(239,68,68,0.2) !important;
    color: #f87171 !important;
}

/* Detection result boxes */
.result-emergency {
    background: rgba(239,68,68,0.15);
    border: 2px solid rgba(239,68,68,0.6);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #f87171;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
}
.result-clear {
    background: rgba(16,185,129,0.15);
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #34d399;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
}
.result-warning {
    background: rgba(245,158,11,0.15);
    border: 2px solid rgba(245,158,11,0.5);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #fbbf24;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
}

/* Pipeline Step */
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
}

/* Info box */
.info-box {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #93c5fd;
    font-size: 0.9rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    try:
        from ultralytics import YOLO
        if os.path.exists("2ndtimebest.pt"):
            return YOLO("2ndtimebest.pt"), None
        return None, "Model file '2ndtimebest.pt' not found."
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_audio_model():
    try:
        import tensorflow as tf
        if os.path.exists("siren_horn_detector.h5"):
            return tf.keras.models.load_model("siren_horn_detector.h5"), None
        return None, "Model file 'siren_horn_detector.h5' not found."
    except Exception as e:
        return None, str(e)


# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🚨 Minor Project — AI/ML</div>
    <div class="hero-title">Emergency Vehicle<br>Detection System</div>
    <div class="hero-subtitle">
        Dual-modal sensor fusion combining YOLOv8 visual detection and 
        deep audio analysis for real-time emergency vehicle identification.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Key Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-value">YOLOv8</div><div class="metric-label">Visual Detection Model</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-value">ANN</div><div class="metric-label">Audio Classification</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-value">2-Modal</div><div class="metric-label">Sensor Fusion</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-value">Real-Time</div><div class="metric-label">Frame Processing</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Demo Video", "🔍 Live Detection", "⚙️ Architecture", "📊 About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demo Video
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🎬 System Demo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        ℹ️ The video below shows the complete fusion system in action — 
        YOLOv8 bounding boxes overlaid on frames with real-time status indicators 
        driven by both visual and audio analysis.
    </div>
    """, unsafe_allow_html=True)

    video_path = "Output_Demo.mp4"
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            st.video(f.read())
    else:
        st.warning("⚠️ `Output_Demo.mp4` not found in the project directory.")

    st.markdown("---")
    st.markdown("### 🚦 Status Legend")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="result-emergency">🔴 !! EMERGENCY !!</div>', unsafe_allow_html=True)
        st.caption("Visual + Audio both detected")
    with col2:
        st.markdown('<div class="result-warning">🟠 WARNING: VISUAL ONLY</div>', unsafe_allow_html=True)
        st.caption("Vehicle seen, no siren")
    with col3:
        st.markdown('<div class="result-warning" style="border-color:rgba(234,179,8,0.5);color:#facc15">🟡 WARNING: SIREN ONLY</div>', unsafe_allow_html=True)
        st.caption("Siren heard, no vehicle seen")
    with col4:
        st.markdown('<div class="result-clear">🟢 SYSTEM: CLEAR</div>', unsafe_allow_html=True)
        st.caption("No emergency detected")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Live Image Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        📸 Upload an image to run live YOLOv8 detection. 
        The model will identify emergency vehicles and draw bounding boxes with confidence scores.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of a road scene to detect emergency vehicles"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(img, use_container_width=True)

        with st.spinner("🔍 Loading YOLOv8 model and running detection..."):
            yolo_model, yolo_err = load_yolo()

        if yolo_err:
            st.error(f"❌ Could not load YOLO model: {yolo_err}")
        elif yolo_model:
            with st.spinner("🚀 Detecting emergency vehicles..."):
                results = yolo_model(img_array, conf=0.4, verbose=False)
                result_img = results[0].plot()  # BGR numpy array
                boxes = results[0].boxes

            with col_result:
                st.markdown("**Detection Result**")
                st.image(result_img[:, :, ::-1], use_container_width=True)  # BGR→RGB

            st.markdown("---")
            if len(boxes) > 0:
                st.markdown('<div class="result-emergency">🚨 EMERGENCY VEHICLE DETECTED!</div>', unsafe_allow_html=True)
                st.markdown(f"<br>**{len(boxes)} detection(s) found:**", unsafe_allow_html=True)
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names.get(cls_id, f"Class {cls_id}")
                    st.markdown(f"- Detection {i+1}: **{cls_name}** — Confidence: `{conf:.2%}`")
            else:
                st.markdown('<div class="result-clear">✅ No Emergency Vehicles Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; background: rgba(255,255,255,0.02); 
             border: 2px dashed rgba(255,255,255,0.1); border-radius: 16px; color: #64748b;">
            <div style="font-size:3rem">📸</div>
            <div style="font-size:1rem; margin-top:0.5rem">Drop an image above to run live detection</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">⚙️ System Architecture</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🔍 Visual Module — YOLOv8")
        st.markdown("""
        <div class="pipeline-step" style="border-color: #f87171;">
            <b style="color:#f87171">Input</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">Video frames extracted at runtime via OpenCV</span>
        </div>
        <div class="pipeline-step" style="border-color: #fb923c;">
            <b style="color:#fb923c">YOLOv8n Model</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">Fine-tuned on emergency vehicle dataset (Roboflow). 
            Detects ambulances, fire trucks, police cars.</span>
        </div>
        <div class="pipeline-step" style="border-color: #facc15;">
            <b style="color:#facc15">Output</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">Bounding boxes + confidence scores → v_detected flag</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔊 Audio Module — ANN")
        st.markdown("""
        <div class="pipeline-step" style="border-color: #60a5fa;">
            <b style="color:#60a5fa">Audio Extraction</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">Librosa extracts 1-second audio chunks at 22,050 Hz sample rate</span>
        </div>
        <div class="pipeline-step" style="border-color: #818cf8;">
            <b style="color:#818cf8">MFCC Features</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">40 Mel-Frequency Cepstral Coefficients extracted per chunk</span>
        </div>
        <div class="pipeline-step" style="border-color: #a78bfa;">
            <b style="color:#a78bfa">ANN Classifier</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">3-layer Dense network trained on UrbanSound8K 
            (classes: Siren & Horn). Output: a_detected flag</span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🔗 Sensor Fusion Engine")
        st.markdown("""
        <div class="pipeline-step" style="border-color: #34d399; margin-top:3.5rem">
            <b style="color:#34d399">Decision Logic</b><br>
            <span style="color:#94a3b8; font-size:0.9rem">
            Combines v_detected + a_detected flags per frame:
            </span>
            <br><br>
            <code style="background:rgba(255,255,255,0.07); padding:0.5rem; border-radius:6px; display:block; font-size:0.82rem">
            if v_detected AND a_detected → EMERGENCY<br>
            elif v_detected → WARNING: VISUAL ONLY<br>
            elif a_detected → WARNING: SIREN ONLY<br>
            else → SYSTEM: CLEAR
            </code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📦 Tech Stack")
        tech = {
            "YOLOv8": ("🔍", "Ultralytics", "#f87171"),
            "TensorFlow/Keras": ("🧠", "Audio ANN", "#60a5fa"),
            "Librosa": ("🔊", "Audio Feature Extraction", "#a78bfa"),
            "OpenCV": ("🎥", "Video Processing", "#34d399"),
            "Streamlit": ("🌐", "Web Interface", "#fb923c"),
        }
        for name, (icon, role, color) in tech.items():
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.75rem; padding:0.6rem 0.8rem; 
                 background:rgba(255,255,255,0.03); border-radius:8px; margin-bottom:0.4rem;
                 border-left: 3px solid {color};">
                <span style="font-size:1.2rem">{icon}</span>
                <div>
                    <span style="font-weight:600; color:{color}">{name}</span>
                    <span style="color:#64748b; font-size:0.8rem; margin-left:0.5rem">— {role}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📊 About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Emergency vehicles (ambulances, fire trucks, police cars) need to navigate 
        through traffic quickly. Traditional systems rely solely on visual cues. 
        This system combines **visual detection** and **audio detection** to create 
        a more reliable, dual-confirmation emergency alert system.
        """)

        st.markdown("### 📁 Dataset")
        st.markdown("""
        | Component | Dataset | Classes |
        |-----------|---------|---------|
        | Visual (YOLO) | Roboflow Emergency Vehicle | Ambulance, Police, Fire Truck |
        | Audio (ANN) | UrbanSound8K | Siren (class 8), Horn (class 1) |
        """)

    with col2:
        st.markdown("### 🏗️ Model Details")
        st.markdown("""
        **YOLOv8 (Visual)**
        - Base: `yolov8n.pt` (Nano)
        - Training: 50 epochs, 640×640 resolution
        - Hardware: GPU (T4 on Google Colab)
        - Output: Bounding boxes + class labels

        **ANN (Audio)**
        - Architecture: `Dense(256) → Dropout → Dense(128) → Dropout → Dense(2)`
        - Input: 40 MFCC features
        - Training: 50 epochs, batch size 32
        - Output: Binary (Siren vs Horn)
        """)

        st.markdown("### ⚙️ How It Works")
        st.markdown("""
        1. **Frame extraction** — OpenCV reads video frame by frame
        2. **Visual inference** — YOLOv8 runs on each frame (conf ≥ 0.5)
        3. **Audio sync** — Librosa processes 1-second audio chunk per second
        4. **Fusion** — Flags from both modules are combined via AND/OR logic
        5. **Overlay** — Status + bounding boxes drawn on output frame
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.85rem; padding:1rem">
        Built with ❤️ using YOLOv8 · TensorFlow · Librosa · Streamlit
    </div>
    """, unsafe_allow_html=True)
