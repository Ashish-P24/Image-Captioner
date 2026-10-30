"""
Digital Image Processing (DIP) Pipeline
Contains image transformation, edge detection, contrast enhancement,
DIP statistical metrics, and spatial attention heatmap generation.
"""

import cv2
import numpy as np
from PIL import Image
import torch


def compute_dip_metrics(image_np: np.ndarray) -> dict:
    """
    Computes classical DIP statistical metrics:
    - Sharpness (Laplacian variance)
    - RMS & Michelson Contrast
    - Shannon Entropy
    - Mean Brightness & Dynamic Range
    - Colorfulness (Hasler & Süsstrunk metric)
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # 1. Spatial Sharpness (Laplacian Variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # 2. Brightness & Dynamic Range
    mean_brightness = float(np.mean(gray))
    min_val, max_val = float(np.min(gray)), float(np.max(gray))
    dynamic_range = max_val - min_val

    # 3. RMS & Michelson Contrast
    rms_contrast = float(np.std(gray))
    michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-8)

    # 4. Shannon Entropy (Information density)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_prob = hist / (hist.sum() + 1e-8)
    hist_prob = hist_prob[hist_prob > 0]
    entropy = -float(np.sum(hist_prob * np.log2(hist_prob)))

    # 5. Colorfulness Index
    if len(image_np.shape) == 3:
        R = image_np[:, :, 0].astype("float")
        G = image_np[:, :, 1].astype("float")
        B = image_np[:, :, 2].astype("float")
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
        mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
        colorfulness = float(std_root + (0.3 * mean_root))
    else:
        colorfulness = 0.0

    return {
        "sharpness": round(sharpness, 2),
        "mean_brightness": round(mean_brightness, 2),
        "dynamic_range": int(dynamic_range),
        "rms_contrast": round(rms_contrast, 2),
        "michelson_contrast": round(float(michelson_contrast), 4),
        "entropy": round(entropy, 3),
        "colorfulness": round(colorfulness, 2),
    }


def apply_dip_transformations(image_np: np.ndarray) -> dict:
    """
    Applies fundamental DIP transformations to demonstrate image operations:
    - Grayscale conversion
    - Histogram Equalization
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Sobel Edge Detection (Gx, Gy, Magnitude)
    - Canny Edge Detection
    - Laplacian Second Derivative
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Sobel Gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.uint8(np.clip((sobel_mag / (sobel_mag.max() + 1e-8)) * 255, 0, 255))

    # Canny Edge Detection
    canny = cv2.Canny(gray, threshold1=100, threshold2=200)

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_uint8 = np.uint8(np.clip(np.abs(laplacian), 0, 255))

    return {
        "gray": gray,
        "hist_eq": hist_eq,
        "clahe": clahe_img,
        "sobel_mag": sobel_mag,
        "canny": canny,
        "laplacian": laplacian_uint8,
    }


def extract_vit_attention_map(model, processor, image: Image.Image) -> np.ndarray:
    """
    Extracts spatial self-attention from the Vision Transformer (ViT) backbone
    of the vision-language model. Computes CLS-token attention to visual patches.
    """
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )

    # attentions: tuple of (batch_size, num_heads, seq_len, seq_len)
    attentions = vision_outputs.attentions
    
    # Aggregate attention across the last 3 transformer layers
    last_layers = attentions[-3:]
    layer_attns = []
    for layer in last_layers:
        # layer shape: (1, num_heads, 577, 577)
        heads_mean = layer[0].mean(dim=0) # (577, 577)
        # Attention from CLS token (index 0) to all patch tokens (index 1:)
        cls_to_patches = heads_mean[0, 1:] # (576,)
        layer_attns.append(cls_to_patches)

    avg_attn = torch.stack(layer_attns).mean(dim=0).detach().cpu().numpy()
    
    # 24x24 patch grid
    grid_size = int(np.sqrt(avg_attn.shape[0]))
    attn_grid = avg_attn.reshape(grid_size, grid_size)

    # Min-max normalization
    attn_normalized = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    # Interpolate to match original image dimensions using Bicubic interpolation
    w, h = image.size
    heatmap_resized = cv2.resize(attn_normalized, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    return heatmap_resized


def generate_heatmap_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    colormap_type: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays spatial attention heatmap on top of the original RGB image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap_type)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1.0 - alpha, colored_heatmap, alpha, 0)
    return overlay
