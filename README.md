# 🚨 Emergency Vehicle Detection System

A dual-modal AI system that detects emergency vehicles by fusing **YOLOv8 visual detection** and **deep audio classification (ANN)** in real time.

## 🔗 Live Demo
> Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🧠 How It Works

| Module | Model | Input | Output |
|--------|-------|-------|--------|
| Visual | YOLOv8n (fine-tuned) | Video frames | Bounding boxes |
| Audio  | ANN (TensorFlow/Keras) | MFCC features | Siren / Horn |
| Fusion | Rule-based logic | Both flags | Status alert |

### Fusion Decision Logic
```
if v_detected AND a_detected → !! EMERGENCY !!
elif v_detected              → WARNING: VISUAL ONLY
elif a_detected              → WARNING: SIREN ONLY
else                         → SYSTEM: CLEAR
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── 2ndtimebest.pt          # YOLOv8 fine-tuned model
├── siren_horn_detector.h5  # Audio ANN model
└── Output_Demo.mp4         # Pre-processed demo video
```

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack
- **YOLOv8** (Ultralytics) — Visual detection
- **TensorFlow / Keras** — Audio ANN
- **Librosa** — MFCC feature extraction
- **OpenCV** — Video frame processing
- **Streamlit** — Web interface

---

## 📊 Datasets
- **Visual**: [Roboflow Emergency Vehicle Dataset](https://roboflow.com)
- **Audio**: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
