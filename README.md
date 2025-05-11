# 🎙️ No Words, Just Tone: Audio-Based Sarcasm Detection

This repository presents our B.Tech research project focused on detecting **sarcasm from audio alone**—without relying on textual or visual inputs. We explore whether acoustic cues in speech can reveal sarcasm effectively, using state-of-the-art pretrained models and deep learning.

> 🧠 Our work demonstrates that sarcasm can be accurately detected from just tone and prosody—challenging the belief that multimodal (text + audio + video) input is necessary.

---

## 📌 Project Highlights

- **Dataset Used**: [MUStARD++](https://paperswithcode.com/dataset/mustard-1) (Multimodal Sarcasm Detection Dataset)
- **Modality Focused**: Audio-only (utterance + context)
- **Pretrained Models**: Wav2Vec2.0, Whisper, HuBERT, LanguageBind, ImageBind, XLS-R, UniSpeech, MMS, xVector, WavLM
- **Model Architectures**:
  - FCN (Fully Connected Network)
  - CNN + FCN
  - Dual-Embedding FCN
  - Dual-Embedding CNN + FCN
- **Performance**: Surpassed multimodal baselines using audio-only features

---

## 🧠 Methodology

### 🗂️ Dataset
We used the **MUStARD++** dataset, which includes short sarcastic/non-sarcastic audio clips from TV shows like *Friends* and *The Office*. For this project, we focused **only on the audio utterance and audio context** clips.

### 🔊 Embedding Extraction
Audio clips were processed using the following **pretrained models** to extract embeddings:

- **Speech models**: Wav2Vec2.0, Whisper, HuBERT, MMS, UniSpeech, WavLM, xVector, XLS-R
- **Multimodal models**: LanguageBind, ImageBind

Each model produced fixed-length embeddings from both utterance and context audio.

### 🏗️ Architectures
We experimented with the following deep learning pipelines:

1. **FCN (Single-Model)**  
   → Dense layers applied to context and utterance embeddings independently and fused for classification.

2. **CNN + FCN (Single-Model)**  
   → Embeddings reshaped and passed through Conv1D layers to capture local sequential audio patterns.

3. **Dual-Embedding FCN**  
   → Embeddings from two different models combined and passed through dense layers.

4. **Dual-Embedding CNN + FCN**  
   → Combines semantic and acoustic strengths of two pretrained models using CNN + FCN fusion layers.

### ⚙️ Training
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score
- **Regularization**: Dropout, BatchNorm, EarlyStopping

---

## 📊 Results Snapshot

| Embedding Type          | Architecture         | Accuracy | F1 Score |
|-------------------------|----------------------|----------|----------|
| Whisper                 | FCN                  | 73.03%   | 73.00%   |
| LanguageBind            | FCN                  | 71.78%   | 71.43%   |
| LanguageBind + XLS-R    | CNN + FCN (Dual)     | **73.86%** | **73.85%** |
| LanguageBind + Whisper  | CNN + FCN (Dual)     | 73.33%   | 73.15%   |

> Dual-model embeddings significantly enhanced performance by capturing both **semantic tone** and **prosodic cues**.

---


