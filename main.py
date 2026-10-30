"""
Digital Image Processing (DIP) Final Project:
Accurate Vision-Language Image Description Generator & Spatial Attention Heatmap Analyzer
Author: Ashish Pathak
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dip_pipeline import (
    apply_dip_transformations,
    compute_dip_metrics,
    extract_vit_attention_map,
    generate_heatmap_overlay,
)
from model_inference import (
    answer_visual_question,
    compute_clip_alignment,
    generate_descriptions,
    load_vision_language_models,
)

# Set page configuration once at root level
st.set_page_config(
    page_title="DIP Image Description Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic presentation styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .caption-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #38bdf8;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .caption-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .caption-text {
        font-size: 1.25rem;
        font-weight: 500;
        color: #ffffff;
        line-height: 1.4;
    }
    .candidate-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
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
    st.markdown('<div class="main-title">🔬 Digital Image Processing: Image Description Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">End-to-End DIP Pipeline: Spatial Filtering, Contrast Enhancement, Vision Transformer Attention Heatmaps & Accurate Multimodal Captioning</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        input_source = st.radio(
            "Select Image Source:",
            ["Preset Sample Image", "Upload Custom Image"],
            index=0,
        )

        image_to_process = None
        if input_source == "Preset Sample Image":
            try:
                image_to_process = Image.open("uploaded_image.jpg")
                st.success("Loaded preset: uploaded_image.jpg")
            except Exception as e:
                st.warning("Preset not found. Please upload an image.")
        else:
            uploaded_file = st.file_uploader(
                "Upload Image (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)

        st.markdown("---")
        st.subheader("🤖 Model Inference Settings")
        beam_size = st.slider("Beam Search Size", min_value=1, max_value=5, value=3)
        max_tokens = st.slider("Max New Tokens", min_value=20, max_value=80, value=45)

        st.markdown("---")
        st.subheader("🗺️ Heatmap Visualization")
        colormap_choice = st.selectbox("Colormap", ["JET", "TURBO", "INFERNO", "VIRIDIS", "MAGMA"], index=0)
        overlay_alpha = st.slider("Heatmap Blending Opacity (α)", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

        st.markdown("---")
        st.caption("Digital Image Processing Final Project | Ashish Pathak")

    if image_to_process is None:
        st.info("👈 Please select or upload an image from the sidebar to begin processing.")
        return

    # Convert to RGB numpy array
    image_rgb = image_to_process.convert("RGB")
    image_np = np.array(image_rgb)

    # Load Vision-Language Models
    with st.spinner("Initializing Deep Vision-Language Models (BLIP & CLIP)..."):
        blip_model, blip_processor, clip_model, clip_processor, device = load_vision_language_models()

    # Create Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Description & Semantic Match",
        "🔬 DIP Steps & Transformations",
        "🗺️ Spatial Attention Heatmaps",
        "💬 Interactive Visual Q&A",
    ])

    # ----------------------------------------------------
    # TAB 1: Description & Semantic Match
    # ----------------------------------------------------
    with tab1:
        st.subheader("Visual Description & Semantic Alignment")
        col_img, col_desc = st.columns([1.1, 1.3])

        with col_img:
            st.image(image_rgb, caption=f"Input Image ({image_rgb.width} × {image_rgb.height})", use_container_width=True)

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

            # Concise Caption
            st.markdown(
                f"""
                <div class="caption-box">
                    <div class="caption-title">📌 Generated Summary Caption</div>
                    <div class="caption-text">{desc_results['concise']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detailed Description
            st.markdown(
                f"""
                <div class="caption-box" style="border-color: #10b981;">
                    <div class="caption-title" style="color: #10b981;">📝 Detailed Multi-Aspect Description</div>
                    <div class="caption-text" style="font-size: 1.05rem;">{desc_results['detailed']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CLIP Semantic Evaluation
            with st.spinner("Computing CLIP multimodal cosine similarity..."):
                sim_score, conf_pct, grade = compute_clip_alignment(
                    image_to_process,
                    desc_results["concise"],
                    clip_model,
                    clip_processor,
                    device,
                )

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("CLIP Cosine Similarity", sim_score)
            with col_m2:
                st.metric("Semantic Match", grade)
            with col_m3:
                st.metric("Model Confidence", f"{conf_pct}%")

            st.progress(int(conf_pct))

        st.markdown("---")
        st.subheader("🎯 Ranked Beam Search Candidates")
        for idx, candidate in enumerate(desc_results["candidates"], start=1):
            cand_score, _, _ = compute_clip_alignment(
                image_to_process,
                candidate,
                clip_model,
                clip_processor,
                device,
            )
            st.markdown(
                f"""
                <div class="candidate-card">
                    <span><b>Candidate {idx}:</b> {candidate}</span>
                    <span style="color: #38bdf8; font-weight: 600;">CLIP Score: {cand_score}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------
    # TAB 2: DIP Steps & Transformations
    # ----------------------------------------------------
    with tab2:
        st.subheader("Digital Image Processing (DIP) Pipeline Execution")
        st.write("Demonstration of classical DIP algorithms applied prior to deep semantic feature extraction:")

        dip_results = apply_dip_transformations(image_np)
        dip_metrics = compute_dip_metrics(image_np)

        # Statistical Metrics Bar
        st.markdown("#### 📊 DIP Quantitative Spatial & Statistical Metrics")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Sharpness (Laplacian Var)", dip_metrics["sharpness"])
        with mcol2:
            st.metric("Shannon Entropy (bits)", dip_metrics["entropy"])
        with mcol3:
            st.metric("RMS Contrast", dip_metrics["rms_contrast"])
        with mcol4:
            st.metric("Mean Brightness", dip_metrics["mean_brightness"])
        with mcol5:
            st.metric("Colorfulness Index", dip_metrics["colorfulness"])

        st.markdown("---")

        # Step 1: Contrast Enhancement
        st.markdown("#### Step 1: Grayscale & Contrast Enhancement")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(dip_results["gray"], caption="1. Grayscale Conversion (Luminance)", use_container_width=True)
        with c2:
            st.image(dip_results["hist_eq"], caption="2. Global Histogram Equalization", use_container_width=True)
        with c3:
            st.image(dip_results["clahe"], caption="3. Adaptive Contrast (CLAHE)", use_container_width=True)

        st.markdown("---")

        # Step 2: Spatial Derivative & Edge Detection
        st.markdown("#### Step 2: Spatial Gradients & Edge Detection Filters")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.image(dip_results["sobel_mag"], caption="1. Sobel Gradient Magnitude (1st Derivative)", use_container_width=True)
        with e2:
            st.image(dip_results["canny"], caption="2. Canny Edge Map (Non-Max Suppression & Hysteresis)", use_container_width=True)
        with e3:
            st.image(dip_results["laplacian"], caption="3. Laplacian High-Pass Filter (2nd Derivative)", use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: Spatial Attention Heatmaps
    # ----------------------------------------------------
    with tab3:
        st.subheader("🗺️ Vision Transformer (ViT) Spatial Attention Heatmap")
        st.write(
            "This visualization reveals the internal spatial self-attention of the Vision Transformer backbone. "
            "Highlighted regions show where the model allocates attention to formulate the linguistic description."
        )

        with st.spinner("Extracting multi-layer Vision Transformer attention maps..."):
            raw_heatmap = extract_vit_attention_map(blip_model, blip_processor, image_to_process)
            overlay = generate_heatmap_overlay(
                image_np,
                raw_heatmap,
                colormap_type=get_colormap(colormap_choice),
                alpha=overlay_alpha,
            )

        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.image(image_rgb, caption="Original Input Image", use_container_width=True)
        with hcol2:
            st.image(
                (raw_heatmap * 255).astype(np.uint8),
                caption="Normalized Patch Attention (Spatial Saliency)",
                use_container_width=True,
            )
        with hcol3:
            st.image(
                overlay,
                caption=f"Heatmap Overlay ({colormap_choice}, α={overlay_alpha})",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown(
            f"""
            > **DIP Explainability Analysis**: The Vision Transformer divides the image into uniform 16×16 spatial patches. 
            The attention rollout above indicates that peak saliency is localized on key semantic focal points, 
            which directly informed the description: *"**{desc_results['concise']}**"*.
            """
        )

    # ----------------------------------------------------
    # TAB 4: Interactive Visual Q&A
    # ----------------------------------------------------
    with tab4:
        st.subheader("💬 Interactive Visual Question Answering (VQA)")
        st.write("Ask targeted questions to probe specific visual attributes, objects, or background elements:")

        quick_questions = [
            "What are the people wearing?",
            "What is the setting or background?",
            "How many people or main objects are in the image?",
            "Describe the colors in the scene.",
        ]

        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button(quick_questions[0]):
                st.session_state["query_input"] = quick_questions[0]
            if st.button(quick_questions[1]):
                st.session_state["query_input"] = quick_questions[1]
        with preset_col2:
            if st.button(quick_questions[2]):
                st.session_state["query_input"] = quick_questions[2]
            if st.button(quick_questions[3]):
                st.session_state["query_input"] = quick_questions[3]

        user_query = st.text_input(
            "Enter your question about the image:",
            value=st.session_state.get("query_input", "What are the subjects doing?"),
        )

        if st.button("Submit Visual Query", type="primary"):
            if user_query.strip():
                with st.spinner("Analyzing image regions and computing answer..."):
                    answer = answer_visual_question(
                        image_to_process,
                        user_query,
                        blip_model,
                        blip_processor,
                        device,
                    )
                st.markdown(
                    f"""
                    <div class="caption-box" style="border-color: #a855f7;">
                        <div class="caption-title" style="color: #a855f7;">💡 Visual Answer</div>
                        <div class="caption-text">{answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()