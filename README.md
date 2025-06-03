# Hybrid Denoising Framework for Remote Sensing Images

This repository contains the official implementation of the manuscript:

**"Hybrid Denoising Framework for Remote Sensing Images: Integrating Spatial Filtering and Deep Learning for Mixed Noise and Shadow Removal"**

Submitted to *The Visual Computer* journal.

## Overview

This hybrid framework removes **mixed noise** (Salt-and-Pepper + White Gaussian Noise) and **shadows** from remote sensing images using a sequential process involving:

- CLAHE: Contrast enhancement in shadowed/bright regions
- MDBUTMF: Salt-and-pepper noise removal using adaptive kernel
- GCF: Gaussian Curvature Filter for structure preservation
- DnCNN: Deep CNN-based restoration for mixed noise
- AMOA: Adaptive Mayfly Optimization Algorithm for filter tuning

The performance is evaluated using PSNR, SSIM, GMSD, MSE, and LPIPS metrics.

---

## Architecture

[Hybrid Denoising Block Diagram](block_diagram.png)


## Files

| File | Description |

| `main.py` | Full implementation of the denoising framework |
| `dncnn_pretrained_mixednoise.keras` | Pretrained DnCNN model |
| `requirements.txt` | Python libraries required |
| `block_diagram.png` | Visual representation of the framework |

##  Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet available, install the key dependencies manually:

```bash
pip install numpy opencv-python scikit-image tensorflow matplotlib tqdm numba lpips torch torchvision
```

## Running the Framework

Modify the paths in `main.py`:

```python
input_dir = r'path_to_your_dataset'
output_dir = r'path_to_save_outputs'
```

Then run:

```bash
python main.py
```

The denoised images and intermediate results will be saved to `output_dir`. Metrics will be printed and averaged across all samples.


## Metrics Evaluated

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **GMSD**: Gradient Magnitude Similarity Deviation
- **MSE**: Mean Squared Error
- **LPIPS**: Learned Perceptual Image Patch Similarity

---

##  Citation

> This repository corresponds to the manuscript submitted to *The Visual Computer*. Please cite the paper if you use this code.


## License

This project is licensed under the MIT License.