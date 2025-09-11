# MSTSGM


This repository provides the official PyTorch implementation of the following paper:

> MSTSGM: A Multi-Scale Temporal-Spatial Guided Model for Image Deblurring
>
> [Bo-Yu Pei], [Ke-Jun Long]*, [Zhi-Bo Gao], [Jian Gu], [Shao-Fei Wang], [Xin-Hu Lu]
>
> In Signal Processing: Image Communication. (* indicates Corresponding author)
>
> Paper: 
>
> Abstract: Image deblurring is a critical task in computer vision, essential for recovering sharp images from blurry ones often caused by motion blur or camera shake. Recent advancements in deep learning have introduced convolutional neural networks (CNNs) as a powerful alternative, enabling the learning of intricate mappings between blurry and sharp images. However, existing deep learning approaches still struggle with effectively capturing low-frequency information and maintaining robustness across diverse blur conditions, while high-frequency details are often inadequately restored due to their susceptibility to motion blur. This paper presents the Multi-Scale Temporal-Spatial Guided Model (MSTSGM), which integrates multi-scale feature decoupling (MSFD), temporal convolution networks (TCN), and edge attention guided reconstruction (EAGR) to enhance deblurring performance. The MSFD captures a wide range of details by decomposing images into multi-scale representations, while the TCN refines these features by modeling temporal dependencies in blur formation. The EAGR focuses on key edge features, effectively improving image clarity. Evaluated on benchmark datasets including GoPro, HIDE, and RealBlur, MSTSGM demonstrates competitive performance, achieving higher PSNR and SSIM metrics compared to state-of-the-art methods. Ablation studies validate the contribution of each component, highlighting the synergistic effects of multi-scale processing, temporal feature integration, and edge attention. Furthermore, MSTSGM's application as a preprocessing step for object detection tasks illustrates its practical utility in enhancing the accuracy of downstream computer vision applications. MSTSGM provides a robust solution for advancing image deblurring and related tasks in the field. Source code is available for research purposes at https://github.com/priplex/MSTSGM.

---

## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Dataset](#Dataset)
3. [Train](#Train)
4. [Test](#Test)
5. [Performance](#Performance)
6. [Acknowledgement](#Acknowledgement)

---

## Dependencies

- Python
- Pytorch (1.8)
  - Different versions may cause some errors.
- scikit-image
- opencv-python
- Tensorboard

---

## Dataset

- Download deblur dataset from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) .

- Unzip files ```dataset``` folder.

- Preprocess dataset by running the command below:

  ``` python data/preprocessing.py```

After preparing data set, the data folder should be like the format below:

```
GOPRO
├─ train
│ ├─ blur    % 2103 image pairs
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ sharp
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ test    % 1111 image pairs
│ ├─ ...... (same as train)

```

---

## Train

To train MSTSGM , run the command below:

``` python main.py --model_name "MSTSGM" --mode "train" --data_dir "dataset/GOPRO" ```


Model weights will be saved in ``` results/model_name/weights``` folder.

---

## Test

To test MSTSGM , run the command below:

``` python main.py --model_name "MSTSGM" --mode "test" --data_dir "dataset/GOPRO" --test_model "MSTSGM.pkl" ```


Output images will be saved in ``` results/model_name/result_image``` folder.

---

## Performance


|   Method    | MSTSGM* | MSTSGM |
| :---------: | :-------: | :--------: |
|  PSNR (dB)  |   33.70   |   33.74    |
|    SSIM     |   0.967   |   0.972    |
| Runtime (s) |   0.019   |   0.028    |

---

## Acknowledgement
The code of MSTSGM is based on [MIMO-UNet](https://github.com/chosj95/MIMO-UNet).


---
