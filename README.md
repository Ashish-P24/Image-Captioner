# Digital Image Processing (DIP) Final Project: Accurate Image Description Generator & Spatial Attention Heatmap Analyzer

**Author:** Ashish Pathak  
**Repository:** [https://github.com/Ashish-P24/Image-Captioner](https://github.com/Ashish-P24/Image-Captioner)

---

## 📌 Project Overview
This project presents an advanced **Digital Image Processing (DIP)** system designed to generate accurate, context-aware visual descriptions of images while visually demonstrating every foundational digital image processing technique and spatial feature representation method that underpins modern computer vision.

Traditional image captioning models (such as toy CNN-LSTMs trained on Flickr8k) suffer from severe hallucinations and generic repetitions. This project addresses those limitations by deploying a state-of-the-art **Vision Transformer (ViT)** vision-language architecture paired with an interactive DIP pipeline that exposes:
1. **Classical DIP transformations and spatial filtering** (Color conversions, Histogram Equalization, CLAHE, Sobel gradients, Canny edges, and Laplacian derivatives).
2. **Quantitative DIP statistical metrics** (Laplacian variance sharpness, Shannon entropy, RMS contrast, and colorfulness).
3. **Visual explainability via Spatial Attention Heatmaps** (ViT patch attention rollout with customizable OpenCV colormaps and opacity blending).
4. **Multimodal semantic evaluation** (OpenAI CLIP cosine similarity alignment).
5. **Interactive Visual Question Answering (VQA)** for targeted region queries.

---

## 🔬 Digital Image Processing (DIP) Methods & Pipeline

### 1. Spatial Preprocessing & Contrast Enhancement
- **Grayscale Conversion (Luminance Transform):**
  $$Y = 0.299R + 0.587G + 0.114B$$
- **Global Histogram Equalization:** Redistributes pixel intensities across the full dynamic range $[0, 255]$ by transforming the cumulative distribution function (CDF).
- **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Divides the image into contextual tiles ($8 \times 8$) and limits contrast amplification to prevent noise amplification in uniform regions.

### 2. Edge Detection & Spatial Gradients
- **Sobel Spatial Derivatives:**
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$
  $$G = \sqrt{G_x^2 + G_y^2}$$
- **Canny Edge Detection:** Multi-stage optimal detector incorporating Gaussian smoothing, gradient calculation, non-maximum suppression (NMS), and dual-threshold hysteresis.
- **Laplacian Filter:** Second-order isotropic derivative operator for detecting rapid intensity variations:
  $$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

### 3. Quantitative Image Quality Metrics
- **Spatial Sharpness ($\sigma^2_{\nabla^2 I}$):** Variance of the Laplacian operator, quantifying the edge sharpness and focus level of the image.
- **Shannon Spatial Entropy ($H$):** Measures the information density and texture complexity:
  $$H = -\sum_{i=0}^{255} P(i) \log_2 P(i)$$
- **RMS Contrast:** Standard deviation of pixel intensities representing overall contrast dispersion.
- **Colorfulness Index (Hasler & Süsstrunk):** Quantifies chromatic variance in opposing color spaces ($R-G$ and $Y-B$).

### 4. Vision Transformer (ViT) Spatial Attention Heatmaps
- The image is decomposed into non-overlapping $16 \times 16$ spatial patches ($24 \times 24 = 576$ patches).
- Patch self-attention matrices $A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ are extracted across transformer layers.
- The attention vector between the `[CLS]` token and all spatial patch tokens is reshaped into a $24 \times 24$ spatial grid.
- Bicubic spatial interpolation projects the attention grid to full image dimensions $(W, H)$, followed by alpha-blending with OpenCV colormaps (`JET`, `TURBO`, `INFERNO`, `VIRIDIS`, `MAGMA`).

### 5. Multimodal Semantic Alignment (CLIP)
- Evaluates the cosine similarity between the image embedding vector $v_I$ and the generated text embedding vector $v_T$:
  $$S_{CLIP} = \frac{v_I \cdot v_T}{\|v_I\| \|v_T\|}$$

---

## 🚀 Running the Project

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10 is installed, then activate the local virtual environment:
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Interactive Streamlit Web Interface
```powershell
streamlit run main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 File Structure
```
├── main.py               # Main Streamlit web application & interactive GUI
├── dip_pipeline.py       # Core DIP algorithms, spatial filters, metrics & heatmaps
├── model_inference.py    # Deep vision-language inference (BLIP, CLIP, VQA)
├── evaluate_metrics.py   # NLP evaluation metrics (BLEU-1..4, ROUGE-L, LCS)
├── requirements.txt      # Python dependencies
├── uploaded_image.jpg    # Sample reference image
└── README.md             # Project documentation & DIP methodology
```

---

## 👤 Author & Collaborator
- **Ashish Pathak** ([@Ashish-P24](https://github.com/Ashish-P24))
