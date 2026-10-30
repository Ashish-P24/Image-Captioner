"""
Vision-Language Model Inference Module
Handles BLIP transformer inference, multi-level image description generation,
beam search candidate ranking, CLIP semantic alignment, and visual Q&A.
"""

from typing import List, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    AutoProcessor,
)
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_vision_language_models():
    """
    Loads BLIP for conditional image description generation
    and OpenAI CLIP for semantic cosine similarity scoring.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    return blip_model, blip_processor, clip_model, clip_processor, device


def generate_descriptions(
    image: Image.Image,
    blip_model,
    blip_processor,
    device: str,
    num_beams: int = 3,
    max_new_tokens: int = 50,
) -> dict:
    """
    Generates multi-level descriptions:
    - Concise Summary Caption
    - Detailed Multi-Aspect Visual Description
    - Ranked Alternative Beam Candidates
    """
    image_rgb = image.convert("RGB")

    # 1. Concise Standard Caption
    inputs = blip_processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out_concise = blip_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
    concise_caption = blip_processor.decode(out_concise[0], skip_special_tokens=True).strip().capitalize()

    # 2. Detailed Multi-Aspect Description
    prompt = "a detailed photography of"
    inputs_detailed = blip_processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_detailed = blip_model.generate(
            **inputs_detailed,
            max_new_tokens=max_new_tokens + 20,
            num_beams=num_beams,
            early_stopping=True,
        )
    detailed_desc = blip_processor.decode(out_detailed[0], skip_special_tokens=True).strip().capitalize()

    # 3. Generate Diverse Alternative Candidates
    with torch.no_grad():
        out_beams = blip_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, 4),
            num_return_sequences=min(num_beams, 4),
            early_stopping=True,
        )
    candidates = []
    for seq in out_beams:
        cand_text = blip_processor.decode(seq, skip_special_tokens=True).strip().capitalize()
        if cand_text not in candidates:
            candidates.append(cand_text)

    return {
        "concise": concise_caption,
        "detailed": detailed_desc,
        "candidates": candidates,
    }


def compute_clip_alignment(
    image: Image.Image,
    caption: str,
    clip_model,
    clip_processor,
    device: str,
) -> Tuple[float, float, str]:
    """
    Computes multimodal semantic alignment using CLIP embeddings:
    Returns (raw_cosine_score, confidence_percentage, semantic_grade)
    """
    image_rgb = image.convert("RGB")
    inputs = clip_processor(
        text=[caption],
        images=image_rgb,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

    cosine_sim = float((image_embeds * text_embeds).sum().item())
    confidence_pct = round(max(0.0, min(100.0, (cosine_sim + 0.1) / 0.5 * 100)), 1)

    if cosine_sim >= 0.32:
        grade = "Excellent Alignment"
    elif cosine_sim >= 0.25:
        grade = "Strong Alignment"
    elif cosine_sim >= 0.18:
        grade = "Moderate Alignment"
    else:
        grade = "Weak Alignment"

    return round(cosine_sim, 4), confidence_pct, grade


def answer_visual_question(
    image: Image.Image,
    question: str,
    blip_model,
    blip_processor,
    device: str,
) -> str:
    """
    Answers questions about specific visual image regions and attributes.
    """
    image_rgb = image.convert("RGB")
    prompt = f"Question: {question} Answer:"
    inputs = blip_processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=2,
            early_stopping=True,
        )
    answer = blip_processor.decode(out[0], skip_special_tokens=True).strip()
    return answer
