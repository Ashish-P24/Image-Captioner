"""
Digital Image Processing (DIP) Final Project:
Accurate Vision-Language Image Description Generator & Spatial Attention Heatmap Analyzer
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dip_pipeline import (
    apply_dip_transformations,
    compute_dip_metrics,
    extract_vit_attention_map,
    generate_gradcam_heatmap,
    generate_heatmap_overlay,
    generate_roi_bounding_boxes,
    generate_spectral_saliency_heatmap,
    render_dip_histogram,
)
from model_inference import (
    compute_clip_alignment,
    generate_descriptions,
    load_vision_language_models,
)

# Page configuration
st.set_page_config(
    page_title="DIP Image Description Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyberpunk / Dark Glassmorphic CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080c14 100%);
        color: #f1f5f9;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Clean, Simple Hero Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.75) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px 0 rgba(56, 189, 248, 0.08);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.25;
    }

    .hero-sub {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 12px 16px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }

    .kpi-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
    }

    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 0.72rem;
        color: #64748b;
    }

    /* Description Capsules */
    .desc-capsule {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.65) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .desc-capsule-emerald {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(6, 78, 59, 0.35) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .capsule-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .capsule-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.45;
    }

    /* Candidate Rows */
    .cand-row {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }

    .cand-row:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(56, 189, 248, 0.3);
    }

    .cand-text {
        font-size: 0.95rem;
        font-weight: 500;
        color: #e2e8f0;
    }

    .cand-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        padding: 0 18px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 41, 59, 0.5);
        color: #f1f5f9;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_colormap(name: str) -> int:
    colormaps = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "MAGMA": cv2.COLORMAP_MAGMA,
    }
    return colormaps.get(name, cv2.COLORMAP_JET)


def main():
    # Clean, simple hero banner (no personal name)
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">Digital Image Processing Laboratory</div>
            <h1 class="hero-title">Image Description & Spatial Heatmap Analyzer</h1>
            <div class="hero-sub">
                End-to-end DIP pipeline combining spatial derivative filtering, contrast enhancement, 
                Grad-CAM saliency localization, and multimodal semantic caption generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Image Input")
        input_source = st.radio(
            "Select Source:",
            ["Preset Reference Sample", "Upload Custom Image"],
            index=0,
        )

        image_to_process = None
        if input_source == "Preset Reference Sample":
            try:
                image_to_process = Image.open("uploaded_image.jpg")
                st.success("Loaded: uploaded_image.jpg")
            except Exception:
                st.warning("Preset not found. Please upload an image.")
        else:
            uploaded_file = st.file_uploader(
                "Upload Image (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)

        st.markdown("---")
        st.markdown("### 🤖 Generation Settings")
        beam_size = st.slider("Beam Search Candidates", min_value=1, max_value=5, value=3)
        max_tokens = st.slider("Max Generation Length", min_value=20, max_value=80, value=50)

        st.markdown("---")
        st.markdown("### 🗺️ Heatmap Controls")
        heatmap_method = st.selectbox(
            "Saliency / Heatmap Method",
            [
                "Grad-CAM Saliency (CNN Feature Activations)",
                "Fourier Spectral Residual (DIP Frequency Saliency)",
                "Vision Transformer Patch Attention",
            ],
            index=0,
        )
        colormap_choice = st.selectbox(
            "Colormap Scheme",
            ["JET", "TURBO", "INFERNO", "VIRIDIS", "MAGMA"],
            index=0,
        )
        overlay_alpha = st.slider("Heatmap Blending (α)", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
        roi_threshold = st.slider("Salient ROI Threshold", min_value=0.3, max_value=0.85, value=0.55, step=0.05)

    if image_to_process is None:
        st.info("👈 Please select or upload an image in the sidebar to begin processing.")
        return

    # Image numpy arrays
    image_rgb = image_to_process.convert("RGB")
    image_np = np.array(image_rgb)

    # Load Deep Models with caching
    with st.spinner("⚡ Initializing Models..."):
        blip_model, blip_processor, clip_model, clip_processor, device = load_vision_language_models()

    # Precompute DIP metrics & transforms
    dip_metrics = compute_dip_metrics(image_np)
    dip_transforms = apply_dip_transformations(image_np)

    # 3 Focused Tabs (Visual Q&A removed as requested)
    tab1, tab2, tab3 = st.tabs([
        "📸 Image Description & Semantic Match",
        "🔬 DIP Steps & Transformations",
        "🗺️ Spatial Heatmap & ROI Studio",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Description & Semantic Match
    # ------------------------------------------------------------------
    with tab1:
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Resolution</div>
                    <div class="kpi-value">{image_rgb.width} × {image_rgb.height}</div>
                    <div class="kpi-sub">Spatial Dimensions</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Laplacian Sharpness</div>
                    <div class="kpi-value">{dip_metrics['sharpness']}</div>
                    <div class="kpi-sub">Edge Focus Metric</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Shannon Entropy</div>
                    <div class="kpi-value">{dip_metrics['entropy']} <span style="font-size:0.85rem;">bits</span></div>
                    <div class="kpi-sub">Information Density</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">RMS Contrast</div>
                    <div class="kpi-value">{dip_metrics['rms_contrast']}</div>
                    <div class="kpi-sub">Dynamic Dispersion</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Colorfulness</div>
                    <div class="kpi-value">{dip_metrics['colorfulness']}</div>
                    <div class="kpi-sub">Chromatic Index</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_img, col_desc = st.columns([1.1, 1.3], gap="large")

        with col_img:
            st.image(image_rgb, caption=f"Original Input ({image_rgb.width} × {image_rgb.height})", use_container_width=True)

        with col_desc:
            with st.spinner("Generating accurate vision-language descriptions..."):
                desc_results = generate_descriptions(
                    image_to_process,
                    blip_model,
                    blip_processor,
                    device,
                    num_beams=beam_size,
                    max_new_tokens=max_tokens,
                )

            # Summary Caption
            st.markdown(
                f"""
                <div class="desc-capsule">
                    <div class="capsule-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">
                        ✦ Primary Summary Caption
                    </div>
                    <div class="capsule-text">{desc_results['concise']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detailed Description
            st.markdown(
                f"""
                <div class="desc-capsule desc-capsule-emerald">
                    <div class="capsule-badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4);">
                        ◈ Detailed Multi-Aspect Description
                    </div>
                    <div class="capsule-text" style="font-size: 1.05rem; color: #e2e8f0;">{desc_results['detailed']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CLIP Alignment
            with st.spinner("Evaluating semantic alignment..."):
                sim_score, conf_pct, grade = compute_clip_alignment(
                    image_to_process,
                    desc_results["concise"],
                    clip_model,
                    clip_processor,
                    device,
                )

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8;">Multimodal Semantic Alignment</span>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #38bdf8;">Cosine Score: {sim_score} ({grade})</span>
                    </div>
                    <div style="background: #1e293b; border-radius: 9999px; height: 10px; overflow: hidden; margin-top: 6px;">
                        <div style="width: {conf_pct}%; background: linear-gradient(90deg, #38bdf8, #10b981); height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #64748b; margin-top: 4px;">
                        <span>0.0 (Unrelated)</span>
                        <span>Confidence: {conf_pct}%</span>
                        <span>0.5+ (Strong Match)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 🎯 Diverse Candidate Captions")
        for idx, candidate in enumerate(desc_results["candidates"], start=1):
            cand_score, cand_conf, _ = compute_clip_alignment(
                image_to_process,
                candidate,
                clip_model,
                clip_processor,
                device,
            )
            st.markdown(
                f"""
                <div class="cand-row">
                    <div class="cand-text"><b>#{idx}</b> &nbsp; {candidate}</div>
                    <div class="cand-score">CLIP: {cand_score} &nbsp; ({cand_conf}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # TAB 2: DIP Transformation Matrix
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("### 🔬 Classical Digital Image Processing Pipeline")
        st.write("Visual progression of spatial filtering, frequency responses, and intensity modifications:")

        # Histogram
        st.markdown("#### 1. Pixel Intensity Distribution (RGB & Luminance)")
        hist_img = render_dip_histogram(image_np)
        st.image(hist_img, use_container_width=True)

        st.markdown("---")

        # Contrast Enhancement
        st.markdown("#### 2. Spatial Contrast Enhancements")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(dip_transforms["gray"], caption="Grayscale Conversion (0.299R + 0.587G + 0.114B)", use_container_width=True)
            st.caption("Standard ITU-R BT.601 luminance decomposition.")
        with c2:
            st.image(dip_transforms["hist_eq"], caption="Global Histogram Equalization", use_container_width=True)
            st.caption("Linearizes CDF across dynamic range [0, 255].")
        with c3:
            st.image(dip_transforms["clahe"], caption="Adaptive Contrast (CLAHE)", use_container_width=True)
            st.caption("Tile-based (8×8) equalization with slope clip limit.")

        st.markdown("---")

        # Spatial Filtering
        st.markdown("#### 3. Spatial Derivatives & High-Pass Edge Detection")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.image(dip_transforms["sobel_mag"], caption="Sobel Gradient Magnitude (1st Derivative)", use_container_width=True)
            st.caption(r"$$G = \sqrt{G_x^2 + G_y^2}$$ detects spatial boundary transitions.")
        with e2:
            st.image(dip_transforms["canny"], caption="Canny Optimal Edge Detector", use_container_width=True)
            st.caption("Gaussian smoothing + NMS + Dual threshold hysteresis.")
        with e3:
            st.image(dip_transforms["laplacian"], caption="Laplacian High-Pass Filter (2nd Derivative)", use_container_width=True)
            st.caption(r"$$\nabla^2 I$$ isolates fine textures and edge zero-crossings.")

    # ------------------------------------------------------------------
    # TAB 3: Spatial Heatmap & ROI Studio
    # ------------------------------------------------------------------
    with tab3:
        st.markdown(f"### 🗺️ Spatial Heatmap & Region of Interest (ROI) Studio")
        st.write(
            f"Active Method: **{heatmap_method}**. "
            "Highlighted regions show where feature activations identify primary semantic focal points in the scene."
        )

        with st.spinner("Computing spatial saliency map..."):
            if "Grad-CAM" in heatmap_method:
                raw_heatmap = generate_gradcam_heatmap(image_to_process)
                explain_text = (
                    "<b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b> computes the gradient of the top visual score "
                    "with respect to convolutional feature maps. This directly localizes the primary semantic object (e.g., the dog, person, or vehicle) "
                    "with high spatial fidelity."
                )
            elif "Spectral" in heatmap_method:
                raw_heatmap = generate_spectral_saliency_heatmap(image_np)
                explain_text = (
                    "<b>Fourier Spectral Residual Saliency</b> is a classical frequency-domain DIP method. It calculates the difference "
                    "between the log amplitude spectrum and its smoothed version, revealing visual novelties and high-contrast regions."
                )
            else:
                raw_heatmap = extract_vit_attention_map(blip_model, blip_processor, image_to_process)
                explain_text = (
                    "<b>Vision Transformer (ViT) Patch Attention</b> shows self-attention aggregated across intermediate transformer layers."
                )

            overlay = generate_heatmap_overlay(
                image_np,
                raw_heatmap,
                colormap_type=get_colormap(colormap_choice),
                alpha=overlay_alpha,
            )
            roi_annotated = generate_roi_bounding_boxes(
                image_np,
                raw_heatmap,
                threshold=roi_threshold,
            )

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.image(image_rgb, caption="1. Original Input Image", use_container_width=True)
        with col_h2:
            norm_heat_display = (raw_heatmap - raw_heatmap.min()) / (raw_heatmap.max() - raw_heatmap.min() + 1e-8)
            st.image(
                (norm_heat_display * 255).astype(np.uint8),
                caption=f"2. Normalized Saliency Map ({heatmap_method.split('(')[0].strip()})",
                use_container_width=True,
            )

        col_h3, col_h4 = st.columns(2)
        with col_h3:
            st.image(
                overlay,
                caption=f"3. Heatmap Overlay ({colormap_choice}, α={overlay_alpha})",
                use_container_width=True,
            )
        with col_h4:
            st.image(
                roi_annotated,
                caption=f"4. Segmented Attention Regions of Interest (ROI ≥ {roi_threshold})",
                use_container_width=True,
            )

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px 20px; margin-top: 14px;">
                <b style="color: #38bdf8;">💡 DIP Explainability Analysis:</b><br>
                {explain_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
