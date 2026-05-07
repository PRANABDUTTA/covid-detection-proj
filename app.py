"""
Chest X-ray classifier — Streamlit UI for the Keras model exported from
Covid19_Chest_Xrays_CNN.ipynb (Task 9).

Place exported files under ./artifacts/ (see README.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Chest X-ray classifier", layout="centered")

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]
IMG_SIZE = (224, 224)
MODEL_FILE = "covid_xray_best_model.keras"
META_FILE = "covid_xray_best_model_metadata.json"


def load_metadata() -> dict:
    p = ART / META_FILE
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    path = ART / MODEL_FILE
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


def preprocess_rgb(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def main():
    meta = load_metadata()
    classes = list(meta["classes"]) if meta.get("classes") else list(CLASS_NAMES)

    st.title("COVID-19 chest X-ray — CNN classifier")
    st.caption(
        "Educational demo only — not for clinical use. "
        "Model exported from `Covid19_Chest_Xrays_CNN.ipynb` (Task 9)."
    )

    model = load_model()
    if model is None:
        st.error(
            f"Missing `{ART / MODEL_FILE}`. Export in Task 9, then copy `.keras` "
            f"(and optional `{META_FILE}`) into `artifacts/` next to this repo layout."
        )
        with st.expander("Expected layout"):
            st.code(
                "med-pred-web/\n"
                "  app.py\n"
                "  artifacts/\n"
                f"    {MODEL_FILE}\n"
                f"    {META_FILE}   # optional\n",
                language="text",
            )
        st.stop()

    if meta:
        st.info(
            f"Metadata: **{meta.get('best_model_name', '?')}** "
            f"(test acc {meta.get('test_acc', '?')}; classes: {classes})."
        )

    up = st.file_uploader("Upload a chest X-ray (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.stop()

    img = Image.open(up)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_rgb(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    st.subheader("Prediction")
    st.metric("Predicted class", classes[pred_idx])

    st.subheader("Class probabilities")
    out_df = pd.DataFrame({"Class": classes, "Probability": probs.astype(float)})
    st.dataframe(out_df, use_container_width=True)

    chart_df = pd.DataFrame({"probability": probs}, index=classes)
    st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
