import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
from PIL import Image
from transformers import CLIPModel, AutoProcessor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


@st.cache_resource
def load_gpt():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return model, tokenizer


@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

import torch.nn.functional as F

def compute_clip_score(image_path, caption):
    model, processor = load_clip()

    image = Image.open(image_path)

    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    # Cosine similarity
    score = (image_embeds * text_embeds).sum().item()

    return round(score, 4)

def select_best_caption(image_path, captions):
    scored = []

    for cap, _ in captions:
        score = compute_clip_score(image_path, cap)
        scored.append((cap, score))

    # sort by score (highest first)
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    best_caption = scored[0]

    return best_caption, scored


st.set_page_config(page_title="Image Caption Generator", layout="wide")


@st.cache_resource
def load_all(model_path, tokenizer_path, feature_extractor_path):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer


def generate_caption(image_path, caption_model, feature_extractor, tokenizer, max_length=34):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    image_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = caption_model.predict([image_features, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)
        if word is None or word == "endseq":
            break

        in_text += " " + word

    caption = in_text.replace("startseq", "").strip()
    return caption, image_features


def beam_search_decoder(image_features, tokenizer, model, max_length=34, beam_width=5):
    sequences = [[[], 0.0]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            in_text = "startseq " + " ".join([tokenizer.index_word.get(i, "") for i in seq])
            seq_input = tokenizer.texts_to_sequences([in_text])[0]
            seq_input = pad_sequences([seq_input], maxlen=max_length)

            yhat = model.predict([image_features, seq_input], verbose=0)[0]
            top = np.argsort(yhat)[-beam_width:]

            for idx in top:
                candidate = [seq + [idx], score - np.log(yhat[idx] + 1e-10)]
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    captions = []
    for seq, score in sequences:
        caption = " ".join([tokenizer.index_word.get(i, "") for i in seq])
        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        captions.append((caption, score))

    return captions


def generate_real_heatmap(image_path, feature_extractor):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(feature_extractor.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [feature_extractor.inputs],
        [feature_extractor.get_layer(last_conv_layer).output, feature_extractor.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    if hasattr(heatmap, "numpy"):
        heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))

    # Overlay
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay

def interpret_clip(score):
    if score > 0.35:
        return "Excellent"
    elif score > 0.25:
        return "Good"
    elif score > 0.15:
        return "Acceptable"
    else:
        return "Poor"


def clip_to_confidence(score):
    return round((score + 1) / 2 * 100, 2)


def diversity_score(captions):
    unique_words = set()
    total_words = 0

    for cap, _ in captions:
        words = cap.split()
        total_words += len(words)
        unique_words.update(words)

    return round(len(unique_words) / total_words, 2)


def consistency_score(scored):
    if len(scored) < 2:
        return 0
    return round(scored[0][1] - scored[1][1], 3)


def main():
    st.title("Image Caption Generator")

    model_path = "image caption model/models/model.keras"
    tokenizer_path = "image caption model/models/tokenizer.pkl"
    feature_extractor_path = "image caption model/models/feature_extractor.keras"

    with st.spinner("Loading models..."):
        caption_model, feature_extractor, tokenizer = load_all(
            model_path, tokenizer_path, feature_extractor_path
        )

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.markdown("---")


        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.image(uploaded_image, use_container_width=True)

        with col2:
            with st.spinner("Generating caption..."):
                caption, image_features = generate_caption(
                    "uploaded_image.jpg", caption_model, feature_extractor, tokenizer
                )

            st.markdown("### Generated Caption")
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#1f2937;">
                    {caption}
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("---")
        st.markdown("### Alternative Captions")

        with st.spinner("Generating alternatives..."):
            captions = beam_search_decoder(image_features, tokenizer, caption_model)

        best_caption, scored_captions = select_best_caption("uploaded_image.jpg", captions)

        # Best caption
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#14532d; color:white;">
                <b>Best Caption:</b><br>{best_caption[0]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Ranked Candidates")

        for i, (cap, score) in enumerate(scored_captions, 1):
            st.markdown(
                f"""
                <div style="padding:10px; margin-bottom:8px; border-radius:8px; background-color:#111827;">
                    <b>{i}.</b> {cap}
                    <span style="float:right;">{round(score,3)}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("---")
        st.markdown("### Heatmap")

        with st.spinner("Generating heatmap..."):
            heatmap = generate_real_heatmap("uploaded_image.jpg", feature_extractor)

        # Center + controlled size
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(heatmap, width=400)

        st.caption("Highlighted regions show where the model focused")


        st.markdown("---")
        st.markdown("### Evaluation")

        with st.spinner("Computing metrics..."):
            clip_score = compute_clip_score("uploaded_image.jpg", caption)

        semantic_label = interpret_clip(clip_score)
        confidence = clip_to_confidence(clip_score)

        # New metrics
        div_score = diversity_score(scored_captions)
        cons_score = consistency_score(scored_captions)
        cap_length = len(caption.split())

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Semantic Match", semantic_label)

        with col2:
            st.metric("Confidence", f"{confidence}%")

        with col3:
            st.metric("Diversity", div_score)

        with col4:
            st.metric("Consistency", cons_score)

        st.markdown("#### Caption Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Length", cap_length)

        with col2:
            st.progress(int(confidence))
        

if __name__ == "__main__":
    main()