<h1 align="center">
  <br/>
  <strong>GreyVive</strong>
</h1>

<p align="center">
  <em>Reviving Grayscale Images with Deep Learning</em><br/>
  A lightweight, production-ready colorization pipeline powered by a U-Net autoencoder and a clean Gradio interface.
</p>

---

## Overview

**GreyVive** is an advanced image colorization framework built with TensorFlow and U-Net architecture. It transforms grayscale images into rich, colorized outputs using the LAB color space. The project is optimized for research, demos, and real-world use cases with a user-friendly Gradio interface.

---

## Features

- **Custom U-Net CNN** for precise color reconstruction
- Trained on **CelebA dataset** of 50K aligned face images
- Color conversion in LAB color space (normalized AB channels)
- Real-time **Gradio UI** with smooth input/output rendering
- Modular codebase for training, preprocessing, and deployment


---


## Installation

Install the project locally:

```bash
git clone https://github.com/Far3s18/GreyVive---Colorize-images.git
cd GreyVive---Colorize-images
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the local interface and test it Gradio:

```bash
python main.py
```

Upload a grayscale image (128x128) or use the example provided.

See the transformation come to life.


## Acknowledgments

- [CelebA Dataset Authors](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
  *Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV 2015.*

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)  
  *Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.*

- [Gradio: Python UI Library for Machine Learning](https://www.gradio.app/)  
  *Gradio Team – Empowering ML apps with clean and shareable UIs.*
