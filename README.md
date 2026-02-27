# MSSA: Memory-Driven and Simplified Scaled Attention for Enhanced Image Captioning (Scientific Reports 2026)

This repository is for **MSSA (Memory-Driven and Simplified Scaled Attention)**, an **LSTM-based** image captioning framework that enhances multimodal integration and caption generation through:

- **Extended Multimodal Feature Extraction** (geometric + color histogram + texture/LBP + edge/Canny + frequency/Gabor descriptors)
- **Memory-Driven Attention (MDA)** with iterative LSTM-memory refinement (**T = 4** by default)
- **Simplified Scaled Attention (SSA)** using **scaled dot-products + bilinear pooling**, while **removing channel-wise gating** for a streamlined attention path inside the LSTM captioning pipeline

Paper (Scientific Reports, 2026): *MSSA: memory-driven and simplified scaled attention for enhanced image captioning*  
DOI: **10.1038/s41598-026-40164-8**  
Project page / code: https://github.com/alamgirustc/MSSA

<p align="center">
  <!-- Replace with your figure path if needed -->
  <img src="images/framework.jpg" width="900"/>
</p>

---

## Citation

Please cite the MSSA paper as:

```bibtex
@article{hossain2026mssa,
  title   = {MSSA: memory-driven and simplified scaled attention for enhanced image captioning},
  author  = {Hossain, Mohammad Alamgir and Ye, ZhongFu and Hossen, Md. Bipul and Rahman, Md. Atiqur and Islam, Md Shohidul and Abdullah, Md. Ibrahim},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-40164-8}
}
```

Baseline reference (X-LAN, CVPR 2020):

```bibtex
@inproceedings{xlinear2020cvpr,
  title={X-Linear Attention Networks for Image Captioning},
  author={Pan, Yingwei and Yao, Ting and Li, Yehao and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

---

## Requirements

- Python 3
- CUDA (GPU recommended)
- numpy, tqdm, easydict
- PyTorch + torchvision
- opencv-python (for Canny edges)
- scikit-image (for LBP and Gabor features)
- coco-caption (for COCO evaluation)

---

## Data preparation

### 1) Bottom-up (region) features
Download the Bottom-Up Attention features and convert them to `.npz` files:

```bash
python2 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_10_100
```

### 2) COCO captions / annotations
Download COCO annotations into the `mscoco/` folder.  
You can follow the data organization used by `self-critical.pytorch`:
- https://github.com/ruotianluo/self-critical.pytorch

### 3) COCO evaluation toolkit
Download **coco-caption** and configure its path (e.g., `C.INFERENCE.COCO_PATH` in your config).

```bash
git clone https://github.com/ruotianluo/coco-caption
```

### 4) Extended multimodal features (MSSA-specific)
In addition to the region visual features, MSSA computes complementary descriptors per ROI and stores them in `.npz` format:

- **Geometric features**: bounding-box and spatial descriptors (**10-D per ROI**)
- **Color histogram**: 3D RGB histogram with **8 bins/channel → 512 bins**
- **Texture (LBP)**: histogram with **256 bins**
- **Edge (Canny)**: histogram with **256 bins**
- **Frequency (Gabor)**: histogram with **256 bins**

In the released configuration, multimodal features are stored with shapes like:
- Geometric: **(52, 10)**
- Color: **(52, 512)**
- Texture/Edge/Gabor: **(52, 256)**

> Notes:
> - The pipeline parses COCO metadata (W, H, filenames), extracts ROIs from bounding boxes, computes each feature type per ROI, normalizes features for scale/resolution invariance, then saves `.npz` files for fast loading.
> - If your ROI count differs (e.g., 10–100), adapt your extraction and padding/truncation strategy to match your model config.

---

## Pretrained checkpoints and generated captions (download)

We provide ready-to-use checkpoints and their corresponding **generated caption results**:

### Cross-Entropy (XE) training
- **Checkpoint (epoch 62):** `caption_model_62.pth`  
  Google Drive: https://drive.google.com/file/d/1oWkS8ixOl5zvGqoYCC0bKgAfz9kYV7Xn/view?usp=sharing
- **Generated captions / test results (epoch 62):** `result_test_62.json`  
  Google Drive: https://drive.google.com/file/d/1LtIBLoFbIkTwPlvs4IWAyFU2UIK3ZQGQ/view?usp=sharing

### RL fine-tuning (CIDEr optimization / SCST)
- **Checkpoint (epoch 25):** `caption_model_25.pth`  
  Google Drive: https://drive.google.com/file/d/1Md7QOSYSt_b4630vv7KFNH8csMOEgdlw/view?usp=sharing
- **Generated captions / test results (epoch 25):** `result_test_25.json`  
  Google Drive: https://drive.google.com/file/d/1YQb35_O6rqefBxwSGAO9_m7ji0NPw6Qh/view?usp=sharing

### Download via command line (recommended)
Install `gdown`:

```bash
pip install -U gdown
```

Download all files into `pretrained/`:

```bash
mkdir -p pretrained

# XE (epoch 62)
gdown --fuzzy "https://drive.google.com/file/d/1oWkS8ixOl5zvGqoYCC0bKgAfz9kYV7Xn/view?usp=sharing" -O pretrained/caption_model_62.pth
gdown --fuzzy "https://drive.google.com/file/d/1LtIBLoFbIkTwPlvs4IWAyFU2UIK3ZQGQ/view?usp=sharing" -O pretrained/result_test_62.json

# RL (epoch 25)
gdown --fuzzy "https://drive.google.com/file/d/1Md7QOSYSt_b4630vv7KFNH8csMOEgdlw/view?usp=sharing" -O pretrained/caption_model_25.pth
gdown --fuzzy "https://drive.google.com/file/d/1YQb35_O6rqefBxwSGAO9_m7ji0NPw6Qh/view?usp=sharing" -O pretrained/result_test_25.json
```

### Where to put the checkpoints
Use the same folder structure as the training scripts:

- XE checkpoint → `experiments/mssa/snapshot/`
- RL checkpoint → `experiments/mssa_rl/snapshot/`

Example:

```bash
cp pretrained/caption_model_62.pth experiments/mssa/snapshot/
cp pretrained/caption_model_25.pth experiments/mssa_rl/snapshot/
```

---

## Training

### Train MSSA (Cross-Entropy)
```bash
bash experiments/mssa/train.sh
```

### Train MSSA using CIDEr optimization / Self-Critical (RL)
Copy the pretrained model into `experiments/mssa_rl/snapshot/` and run:

```bash
bash experiments/mssa_rl/train.sh
```

---

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --folder experiments/model_folder --resume model_epoch
```

---

## Implementation details (paper defaults)

- **Dataset split**: COCO Karpathy split (113,287 train / 5,000 val / 5,000 test)
- **Caption preprocessing**: lowercase, remove words appearing fewer than 6 times (vocab size 9,487)
- **Default MDA refinement iterations**: **T = 4**
- **Two-stage optimization**: Cross-Entropy training followed by CIDEr-optimized fine-tuning
- **Reproducibility**: fixed random seed across Python/NumPy/PyTorch and deterministic CuDNN settings (see paper for the exact seed and full hyperparameter table)

---

## Acknowledgements

Thanks to the contribution of:
- `self-critical.pytorch`: https://github.com/ruotianluo/self-critical.pytorch
- Bottom-Up Attention features: https://github.com/peteanderson80/bottom-up-attention
- The PyTorch community

---

## License

Please follow the license terms of any third-party resources (COCO, coco-caption, bottom-up features, etc.).  
The Scientific Reports article is Open Access under **CC BY-NC-ND 4.0** (see the paper for details).
