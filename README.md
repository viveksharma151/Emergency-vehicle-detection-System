# Emergency Vehicle Detection System

A minor project that detects emergency vehicles using a combination of visual and audio analysis. It uses YOLOv8 for object detection in video frames, and a trained ANN to classify siren/horn sounds from the audio. Both outputs are fused to give a final alert status.

## Live Demo
Deployed on Streamlit Community Cloud.

---

## How It Works

The system processes a video in two parallel ways:

- **Visual**: YOLOv8 (fine-tuned on a Roboflow emergency vehicle dataset) scans each frame and flags if an ambulance, fire truck, or police car is visible.
- **Audio**: Librosa extracts 1-second MFCC features from the video audio, which are fed into a small TensorFlow ANN trained on UrbanSound8K (siren and horn classes).
- **Fusion**: Both flags are combined each second using simple rule-based logic:

```
v_detected + a_detected  =>  EMERGENCY
v_detected only          =>  WARNING: VISUAL ONLY
a_detected only          =>  WARNING: SIREN ONLY
neither                  =>  SYSTEM: CLEAR
```

---

## Project Structure

```
app.py                  - main Streamlit app
requirements.txt        - dependencies
2ndtimebest.pt          - fine-tuned YOLOv8 model weights
siren_horn_detector.h5  - trained audio ANN
Output_Demo.mp4         - sample output video
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- YOLOv8 (Ultralytics) for visual detection
- TensorFlow/Keras for the audio classifier
- Librosa for MFCC feature extraction
- OpenCV for frame-by-frame video processing
- Streamlit for the web UI

---

## Datasets Used

- Visual model: [Roboflow Emergency Vehicle Dataset](https://roboflow.com)
- Audio model: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
