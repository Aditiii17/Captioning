
---

# 🖼️ Image Captioning with Attention Mechanism

This project implements an **attention-based image captioning model** using **PyTorch**, where the goal is to generate natural language descriptions for images. The model uses **ResNet50** for feature extraction and an **LSTM with attention** to generate captions word-by-word.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Model Architecture](#model-architecture)
* [Key Components](#key-components)
* [How It Works](#how-it-works)
* [Setup](#setup)
* [Training](#training)
* [Results](#results)
* [Credits](#credits)

---

## ✅ Overview

Image captioning is a challenging task that combines **computer vision** and **natural language processing**. This implementation:

* Extracts features from images using **ResNet50**
* Processes captions using a **custom Vocabulary class**
* Trains a model using **LSTM + Attention**
* Uses an **adaptive training loop** with optimizations like `AMP`, `Teacher Forcing`, and `Early Stopping`

---

## 🧠 Model Architecture

### 1. **Feature Extractor (Encoder)**

* **Model**: `ResNet50` (pretrained on ImageNet)
* **Use**: Extracts deep feature representations from input images
* **Why**: CNNs like ResNet are excellent at capturing visual patterns and spatial hierarchies

### 2. **Attention-Based Decoder (Decoder)**

* **Model**: `LSTM` with **Bahdanau-style additive attention**
* **Use**: Generates captions word-by-word while focusing on relevant parts of the image
* **Why**: Attention helps align the words being generated with visual regions, improving accuracy and interpretability

---

## 🧩 Key Components

### `Vocabulary` class

* Converts words ↔ indices
* Filters rare words using `frequency_threshold`
* Used to tokenize and detokenize captions

### `load_captions()`

* Loads and parses captions from dataset files
* Handles special format: `image_name| comment_number| comment`

### `ImageCaptionDataset` class

* A PyTorch `Dataset` to load image-caption pairs
* Applies transforms (resize, normalize, etc.)
* Used by `DataLoader` for batch training

### `Attention` module

* Computes attention scores for image features
* Guides the decoder to focus on the most relevant regions

### `Encoder` class

* Modifies ResNet to output spatial feature maps
* Removes classification layer and freezes initial layers

### `Decoder` class

* Combines attention-weighted features and embeddings
* Uses LSTM to sequentially predict words
* Applies linear layer + softmax for output vocabulary prediction

---

## ⚙️ How It Works

1. **Preprocessing**

   * Resize and normalize images
   * Tokenize and index words in captions

2. **Feature Extraction**

   * Feed images through ResNet50 → get 2048-dim feature maps (1×2048×7×7)

3. **Caption Generation**

   * At each timestep:

     * Attention is computed over image features
     * Context vector and word embedding are fed to LSTM
     * Output is mapped to vocabulary to predict the next word

4. **Loss Computation**

   * CrossEntropyLoss over non-`<PAD>` tokens
   * Supports Teacher Forcing

5. **Training Loop**

   * Mixed precision (`autocast`)
   * Gradient scaling with `GradScaler`
   * Early stopping based on validation loss

---

## 🛠️ Setup

### Install Dependencies

```bash
pip install torch torchvision nltk tqdm
```

### Download NLTK Tokenizer

```python
import nltk
nltk.download('punkt')
```

---

## 🏋️‍♀️ Training

Modify hyperparameters and run the training script:

```bash
python train.py
```

### Key Args

* `embed_size`: Dimensionality of word embeddings
* `hidden_size`: Hidden dimension of LSTM
* `attention_dim`: Size of attention vector
* `batch_size`, `epochs`, `learning_rate`: Training controls

---

## 📈 Results

* Dataset: Flickr8k or custom pipe-separated caption dataset
* Model produces grammatically coherent and visually relevant captions
* Attention visualization is possible (not included here)

---

## 🙌 Credits

* **Model Design**: Based on the Show, Attend and Tell paper
* **Libraries Used**: PyTorch, torchvision, NLTK
* **Author**: \[Your Name]
* **Inspired by**: Projects from Stanford CS231n, OpenAI, and image-captioning research

---

## 🗃️ Project Structure

```bash
📦image-captioning
 ┣ 📜train.py                # Main training script
 ┣ 📜vocab.pkl               # Saved vocabulary
 ┣ 📜caption_model.pth       # Trained model weights
 ┣ 📂images/                 # Raw input images
 ┣ 📂captions/               # Pipe-separated caption files
 ┗ 📜README.md               # Project documentation
```

