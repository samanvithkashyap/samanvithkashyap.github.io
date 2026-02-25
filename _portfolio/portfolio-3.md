---
title: "ArtExtract"
excerpt: "A CRNN model for artistic style classification trained on 81,444 WikiArt paintings.<br/><img src='/images/outliers-visualization.png' style='max-width:100%;height:180px;object-fit:cover;border-radius:4px;margin-top:8px;'>"
collection: portfolio
---

**[View Repository on GitHub](https://github.com/samanvithkashyap/wikiart-artextract)**

ArtExtract is a CRNN (Convolutional-Recurrent Neural Network) trained on 81,444 paintings across 29 artistic styles from the WikiArt dataset. Built as a baseline for the **HumanAI ArtExtract GSoC 2026** task — the foundation toward detecting hidden paintings and anomalies in art. Style isn't a local feature, so unlike a plain CNN, the BiGRU layer captures left-to-right and right-to-left relationships across the full canvas width — exactly what style recognition needs.

### Key Technologies
* **Backbone:** EfficientNet-B3 + 2-layer Bidirectional GRU
* **Language:** Python
* **Framework:** PyTorch
* **Concepts Applied:** CRNN architecture, outlier detection, compound scaling, multi-task learning
