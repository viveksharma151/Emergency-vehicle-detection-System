# Emergency Vehicle Detection System

A minor project built to detect emergency vehicles using both visual and audio signals at the same time. Instead of relying on just camera footage, it also listens for sirens and horns — and combines both to decide whether an emergency vehicle is actually present.

**Live Demo:** [emergency-vehicle-detection-system-oh9dtszn2qb7fqnafcfsqr.streamlit.app](https://emergency-vehicle-detection-system-oh9dtszn2qb7fqnafcfsqr.streamlit.app)

---

## What it does

- Runs YOLOv8 on each video frame to spot ambulances, fire trucks, and police cars
- Extracts audio from the video using Librosa and classifies it (siren / horn) with a trained ANN
- Combines both results every second using simple logic:

```
visual + audio detected  →  EMERGENCY
visual only              →  WARNING: VISUAL ONLY
audio only               →  WARNING: SIREN ONLY
neither                  →  CLEAR
```

---

## Project files

```
app.py                   main Streamlit app
requirements.txt         Python dependencies
2ndtimebest.pt           YOLOv8 fine-tuned weights
siren_horn_detector.h5   audio ANN weights
Output_Demo.mp4          sample output video
```

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

- YOLOv8 (Ultralytics) — object detection
- h5py + NumPy — audio model inference
- Librosa — MFCC feature extraction
- OpenCV — video frame processing
- Streamlit — web UI

---

## Datasets

- Visual model trained on: [Roboflow Emergency Vehicle Dataset](https://roboflow.com)
- Audio model trained on: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
