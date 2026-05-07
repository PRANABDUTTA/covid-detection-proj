"""
Chest X-ray classifier — Streamlit UI for the Keras model exported from
Covid19_Chest_Xrays_CNN.ipynb (Task 9).

Place exported files under ./artifacts/ (see README.md).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
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

# Streamlit Cloud: repo is often deployed without the .keras file (gitignored).
# Set MODEL_URL in App settings → Secrets (or env) to a direct HTTPS URL to the file.
_CACHE_DIR = Path(tempfile.gettempdir()) / "covid_xray_streamlit_cache"


def _model_url() -> str | None:
    u = os.environ.get("MODEL_URL", "").strip()
    if u:
        return u
    try:
        if "MODEL_URL" in st.secrets:
            return str(st.secrets["MODEL_URL"]).strip()
    except (FileNotFoundError, KeyError, RuntimeError):
        pass
    return None


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "covid-xray-streamlit/1.0"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        with open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)


def get_model_path() -> tuple[Path | None, str | None]:
    """Return path to .keras file, or (None, error message)."""
    local = ART / MODEL_FILE
    if local.exists():
        return local, None

    url = _model_url()
    if not url:
        return None, None

    cached = _CACHE_DIR / MODEL_FILE
    try:
        if not cached.exists():
            _download_file(url, cached)
    except Exception as e:
        return None, f"Could not download model from MODEL_URL: {e}"

    return cached, None


def load_metadata() -> dict:
    p = ART / META_FILE
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path_str: str):
    return tf.keras.models.load_model(Path(path_str))


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

    path, dl_err = get_model_path()
    if dl_err:
        st.error(dl_err)
        st.stop()

    if path is None:
        st.error(
            f"Missing `{ART / MODEL_FILE}` in the deployed repo. "
            "Export in Task 9, then either:\n\n"
            "**A)** Commit the file under `artifacts/` (use **Git LFS** if >100 MB), **or**\n\n"
            "**B)** Host the `.keras` file at a **direct download HTTPS URL** and add to "
            "**App settings → Secrets**:\n\n"
            "`MODEL_URL = \"https://.../covid_xray_best_model.keras\"`"
        )
        with st.expander("Expected layout (option A)"):
            st.code(
                "med-pred-web/\n"
                "  app.py\n"
                "  artifacts/\n"
                f"    {MODEL_FILE}\n"
                f"    {META_FILE}   # optional\n",
                language="text",
            )
        st.stop()

    model = load_model(str(path))

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
