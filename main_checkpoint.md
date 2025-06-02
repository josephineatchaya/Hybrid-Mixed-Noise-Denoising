# Hybrid Denoising Framework for Remote Sensing Images

This repository contains the implementation and pretrained model for the manuscript:

**"Hybrid Denoising Framework for Remote Sensing Images: Integrating Spatial Filtering and Deep Learning for Mixed Noise and Shadow Removal"**

Submitted to *The Visual Computer* journal.

---

## 🧠 Overview

This hybrid framework addresses denoising of remote sensing images affected by mixed noise (salt-and-pepper + white Gaussian noise) and shadow artifacts. The pipeline combines:

- **CLAHE** for contrast enhancement in shadowed regions
- **MDBUTMF** to suppress salt-and-pepper noise
- **GCF** for structure-preserving smoothing
- **DnCNN** (deep CNN) for fine detail and texture restoration
- **AMOA** for adaptive parameter tuning based on image luminance

All components are integrated into a single script: `main.py`.

---

## 📂 Files Included

| File                        | Description                                 |
|-----------------------------|---------------------------------------------|
| `main.py`                   | Full hybrid denoising pipeline              |
| `pretrained_dncnn_model.h5` | Pretrained DnCNN weights (mixed noise)      |
| `requirements.txt`          | Python dependencies                         |
| `LICENSE`                   | MIT license                                 |
| `sample_data/` *(optional)* | Demo image for quick testing                |

---

## ⚙️ Installation

Install all dependencies:

```bash
pip install -r requirements.txt
