# ECAPA Acoustic Domain Classifier

### Subtitle
**Speech, Music, and Noise Classification Using ECAPA-TDNN Embeddings**

---   

## 🧠 Overview
This model classifies short audio clips into **Speech**, **Music**, or **Noise** domains.  
It uses **ECAPA-TDNN embeddings**, a neural architecture optimized for speaker and acoustic feature representation.

Despite being trained on a **small, human-curated dataset (5 samples per class)**, the model demonstrates **high robustness and near-perfect classification**.  
This project serves as a **proof-of-concept** highlighting how ECAPA embeddings can generalize even in limited-data scenarios.

---

## 📦 Model Information

- **Architecture:** ECAPA-TDNN
- **Framework:** PyTorch (SpeechBrain-based)
- **Input:** Mono audio waveform (16 kHz sampling rate)
- **Output Classes:** Speech | Music | Noise
- **Training Data:** 15 samples (5 per class), normalized and balanced
- **Accuracy:** 100% on internal validation (small-scale)
- **Author:** Khubaib Ahmad — AI/ML Engineer, Data Scientist

---

## ⚙️ Methodology

1. Extract ECAPA-TDNN embeddings for all samples using SpeechBrain.  
2. Train a simple classifier (e.g., linear or small dense network) on embeddings.  
3. Validate predictions using held-out data.  
4. Export trained model weights as `.pkl` file.  

---

## 🚀 Usage Example

```python
import joblib
from speechbrain.pretrained import EncoderClassifier
import torch
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Load trained logistic regression model
loaded_model = joblib.load("custom_audio_classifier.pkl")
print("Model loaded successfully")

# Load ECAPA embedding extractor
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Prediction function
def prediction(filename, clf_name):
    audio = filename
    clf = clf_name
    signal = classifier.load_audio(audio)
    embd = classifier.encode_batch(signal).detach().cpu().numpy().mean(axis=1)
    prediction = clf.predict(embd)
    proba = clf.predict_proba(embd)
    
    print("Predicted Class:", prediction[0])
    print("Class order:", clf.classes_)
    print("Class Probabilities:", proba[0])

# Example inference
prediction(filename="A_voice.opus", clf_name=loaded_model) #-. music/ noise/ speech
```

---

## 📂 File Information

| File | Description |
|------|--------------|
| `ECAPA_acoustic_domain_classifier.pkl` | Trained model weights |
| `requirements.txt` | Dependencies for inference |
| `README.md` | Model documentation |
| `example_audio.mp3` | Sample audio file |

---

## 📊 Applications

- Acoustic scene classification  
- Pre-filtering for speech recognition pipelines  
- Smart audio event detection  
- Sound domain separation tasks

---

## 🔖 Suggested Citation

```
Muhammad Khubaib Ahmad (2025). ECAPA Acoustic Domain Classifier: Differentiating Speech, Music, and Noise using ECAPA-TDNN Embeddings. Hugging Face.
```

---

## 🧾 License
MIT License — free for research and educational use.
