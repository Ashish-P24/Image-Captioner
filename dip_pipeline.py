"""
Digital Image Processing (DIP) Pipeline
Contains image transformation, edge detection, contrast enhancement,
DIP statistical metrics, Grad-CAM spatial heatmaps, and spectral saliency.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import streamlit as st


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
    Applies fundamental DIP transformations:
    - Grayscale conversion
    - Histogram Equalization
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Sobel Gradients (Gx, Gy, Magnitude)
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
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
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


@st.cache_resource(show_spinner=False)
def load_gradcam_model():
    """
    Loads pretrained convolutional feature extractor for Grad-CAM spatial localization.
    """
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    last_conv_layer_name = "Conv_1"
    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )
    return grad_model


def generate_gradcam_heatmap(image: Image.Image) -> np.ndarray:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    highlighting the primary semantic object that drives visual recognition.
    """
    grad_model = load_gradcam_model()
    orig_w, orig_h = image.size
    
    img_resized = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_preprocessed)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    cam = conv_outputs @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.math.reduce_max(cam) + 1e-8)
    cam_np = cam.numpy()

    # Interpolate to match original image dimensions
    heatmap_resized = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    return heatmap_resized


def extract_vit_attention_map(model, processor, image: Image.Image) -> np.ndarray:
    """
    Extracts spatial self-attention from the Vision Transformer (ViT) backbone.
    """
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )

    attentions = vision_outputs.attentions
    
    # Average across middle-to-late transformer layers
    target_layers = attentions[4:10]
    layer_attns = []
    for layer in target_layers:
        heads_mean = layer[0].mean(dim=0) # (577, 577)
        cls_to_patches = heads_mean[0, 1:] # (576,)
        layer_attns.append(cls_to_patches)

    avg_attn = torch.stack(layer_attns).mean(dim=0).detach().cpu().numpy()
    grid_size = int(np.sqrt(avg_attn.shape[0]))
    attn_grid = avg_attn.reshape(grid_size, grid_size)

    attn_normalized = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    w, h = image.size
    heatmap_resized = cv2.resize(attn_normalized, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    return heatmap_resized


def generate_spectral_saliency_heatmap(image_np: np.ndarray) -> np.ndarray:
    """
    Classical DIP Frequency-Domain Saliency via Fourier Spectral Residual (Hou & Zhang).
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    orig_h, orig_w = gray.shape
    scale = 256
    gray_small = cv2.resize(gray, (scale, scale)).astype(np.float32)

    fft = np.fft.fft2(gray_small)
    amplitude = np.abs(fft)
    log_amplitude = np.log(amplitude + 1e-6)
    phase = np.angle(fft)

    spectral_residual = log_amplitude - cv2.boxFilter(log_amplitude, -1, (7, 7))
    combined = np.exp(spectral_residual + 1j * phase)
    ifft = np.fft.ifft2(combined)
    saliency = np.abs(ifft) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)

    saliency_full = cv2.resize(saliency, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    saliency_norm = (saliency_full - saliency_full.min()) / (saliency_full.max() - saliency_full.min() + 1e-8)
    return np.clip(saliency_norm, 0.0, 1.0)


def generate_heatmap_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    colormap_type: int = cv2.COLORMAP_JET,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlays spatial attention heatmap on top of the original RGB image.
    """
    # Normalize to [0, 255]
    norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * norm_heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap_type)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1.0 - alpha, colored_heatmap, alpha, 0)
    return overlay


def generate_roi_bounding_boxes(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = 0.60,
) -> np.ndarray:
    """
    Extracts high-salience attention clusters and draws glowing bounding boxes (ROI)
    around primary semantic visual focal points.
    """
    annotated = image_np.copy()
    norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    mask = np.uint8((norm_heatmap >= threshold) * 255)
    
    # Morphological closing & dilation to consolidate detection regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_dilated = cv2.dilate(mask_closed, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    box_count = 0
    img_area = image_np.shape[0] * image_np.shape[1]
    
    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (img_area * 0.02):  # At least 2% of image
            x, y, w, h = cv2.boundingRect(cnt)
            box_count += 1
            # Draw double border for high-tech highlight
            cv2.rectangle(annotated, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 242, 254), 2)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (56, 189, 248), 2)
            cv2.putText(
                annotated,
                f"ROI #{box_count}",
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 242, 254),
                2,
                cv2.LINE_AA,
            )
            if box_count >= 3:
                break

    return annotated


def render_dip_histogram(image_np: np.ndarray) -> np.ndarray:
    """
    Generates an aesthetic dark-themed RGB + Luminance histogram plot.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    if len(image_np.shape) == 3:
        colors = [("#ef4444", "Red Channel"), ("#22c55e", "Green Channel"), ("#3b82f6", "Blue Channel")]
        for idx, (col, label) in enumerate(colors):
            hist = cv2.calcHist([image_np], [idx], None, [256], [0, 256])
            ax.plot(hist, color=col, linewidth=1.5, alpha=0.85, label=label)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax.plot(hist_gray, color="#e2e8f0", linewidth=2.0, linestyle="--", alpha=0.9, label="Luminance (Gray)")

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity (0 - 255)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Pixel Frequency", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    hist_plot = Image.open(buf)
    return np.array(hist_plot)