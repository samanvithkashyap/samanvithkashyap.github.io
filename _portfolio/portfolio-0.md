# WikiArt Style Classification — ArtExtract GSoC 2026

A CRNN (Convolutional-Recurrent Neural Network) trained on 81,444 paintings across 29 artistic styles from the WikiArt dataset. Built as a baseline for the **HumanAI ArtExtract GSoC 2026** task — the foundation toward detecting hidden paintings and anomalies in art.

---

## Overview

Before you can find hidden paintings or detect anomalies in art, you need a model that genuinely understands artistic style. That's what this is.

The core insight: style isn't a local feature. What tells you a painting is Impressionist isn't any single patch — it's the way loose brushstrokes repeat and flow across the entire canvas. A plain CNN misses that. This architecture doesn't.

---

## Architecture

**Backbone:** EfficientNet-B3 → 1536-channel feature maps  
**Sequential layer:** Reshape to column sequence → 2-layer Bidirectional GRU (hidden=256)  
**Classifier:** FC(512 → 29 styles)

**Why CRNN?**  
The BiGRU treats the image as a sequence of vertical column features extracted by EfficientNet, capturing left-to-right and right-to-left relationships across the full canvas width. That sequential context is exactly what style recognition needs.

**Why EfficientNet-B3?**  
Compound scaling (width + depth + resolution together) gives richer feature maps with fewer parameters than ResNet or VGG. For art images where texture and fine detail matter, that efficiency translates directly to better feature extraction.

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | WikiArt — 81,444 images, 29 styles, 195 artists, 10 genres |
| Training hardware | Kaggle T4 GPU |
| Initial loss | 1.69 |
| Loss after 5 epochs | 0.64 |

---

## Outlier Detection

An outlier detection pipeline using per-image cross-entropy loss + confidence scoring identifies paintings that don't visually fit their assigned style label.

**Logic:**
- High loss + high confidence → genuine outlier (model is certain it belongs elsewhere)
- High loss + low confidence → hard example (model is just uncertain)

**Notable findings:**

- **Jackson Pollock** (labeled Abstract Expressionism) → predicted Baroque at 99% confidence. Drip paintings look unlike anything else in the dataset — the model had no reference frame.
- **Hilma af Klint** (labeled Symbolism) → predicted Minimalism. She painted abstract geometric works in 1906, decades before abstraction existed as a movement. The label is arguably wrong.
- **Edward Hopper** (labeled New Realism) → predicted Realism. Likely a dataset mislabeling; Hopper is commonly miscategorized in WikiArt.

![Outlier Visualization](./images/outliers_visualization.png)

---

## Roadmap — Multi-Task Architecture

The next version extends this to a multi-head model predicting Style, Artist, and Genre simultaneously:

- Shared EfficientNet-B3 backbone
- **Style head:** BiGRU (spatial sequence matters for brushstroke patterns)
- **Artist head:** Global Average Pooling → FC layers
- **Genre head:** Global Average Pooling → FC layers
- **Combined loss:** λ₁·L_style + λ₂·L_artist + λ₃·L_genre

Training is two-phase: freeze backbone first, then fine-tune last 3 blocks at a lower learning rate.

---

## Repository Structure

```
wikiart-artextract/
├── train-initial-architechure.ipynb   # Full training + outlier detection pipeline
├── outliers_visualization.png         # Outlier analysis output
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision efficientnet-pytorch kaggle
```

### Dataset

Download the WikiArt dataset via Kaggle:

```bash
kaggle datasets download -d ikarus777/best-artworks-of-all-time
```

Or use the full WikiArt dataset (81k images) directly from Kaggle notebooks where it's available as a dataset input.

### Run

Open `train-initial-architechure.ipynb` in Kaggle or locally with Jupyter:

```bash
jupyter notebook train-initial-architechure.ipynb
```

---

## Background

This project is a submission baseline for the [HumanAI ArtExtract GSoC 2026](https://humanai.foundation/) task, which focuses on building computer vision systems capable of analyzing fine art — including detecting hidden underpaintings, style classification, and anomaly detection in museum collections.

---

## Author

**Samanvith Kashyap** — [GitHub](https://github.com/samanvithkashyap) · [Portfolio](https://samanvithkashyap.github.io)
