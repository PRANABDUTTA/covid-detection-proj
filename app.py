"""
Chest X-ray classifier — Streamlit UI for the Keras model exported from
Covid19_Chest_Xrays_CNN.ipynb (Task 9).

Place exported files under ./artifacts/ (see README.md).
"""

from __future__ import annotations

import hashlib
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

# .keras is a ZIP archive; HTML error pages often start with "<".
_KERAS_MAGIC = b"PK\x03\x04"


def _looks_like_keras_file(path: Path) -> bool:
    try:
        if path.stat().st_size < 4096:
            return False
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False
    if head.lstrip().startswith(b"<") or head.lstrip().startswith(b"<!"):
        return False
    return head.startswith(_KERAS_MAGIC)


def _cached_model_path_for_url(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{h}_{MODEL_FILE}"


def _model_url() -> str | None:
    u = os.environ.get("MODEL_URL", "").strip()
    if u:
        return u
    for key, val in os.environ.items():
        if key.upper() == "MODEL_URL" and val.strip():
            return val.strip()
    try:
        if "MODEL_URL" in st.secrets:
            return str(st.secrets["MODEL_URL"]).strip()
    except (FileNotFoundError, KeyError, RuntimeError):
        pass
    return None


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "covid-xray-streamlit/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status and int(status) >= 400:
                raise OSError(f"HTTP {status} from URL (check link / permissions).")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def get_model_path() -> tuple[Path | None, str | None]:
    """Return path to .keras file, or (None, error message)."""
    local = ART / MODEL_FILE
    if local.exists():
        return local, None

    url = _model_url()
    if not url:
        return None, None

    cached = _cached_model_path_for_url(url)
    try:
        if cached.exists() and _looks_like_keras_file(cached):
            return cached, None
        if cached.exists():
            cached.unlink()

        _download_file(url, cached)
        if not _looks_like_keras_file(cached):
            cached.unlink(missing_ok=True)
            return (
                None,
                "MODEL_URL downloaded something that is not a valid `.keras` ZIP "
                "(often an HTML login or error page). Use a **direct file** URL — e.g. "
                "GitHub **Release asset** link, not the repo file viewer page.",
            )
    except Exception as e:
        cached.unlink(missing_ok=True)
        return None, f"Could not download model from MODEL_URL: {e}"

    return cached, None


def load_metadata() -> dict:
    p = ART / META_FILE
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path_str: str):
    # compile=False avoids optimizer edge cases and matches inference-only use.
    return tf.keras.models.load_model(Path(path_str), compile=False)


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
        expected = ART / MODEL_FILE
        st.error(
            "**No model available.** The app checks two things in order:\n\n"
            "1. **File in Git next to `app.py`:**\n"
            f"   `{expected}`\n\n"
            "2. **`MODEL_URL`** in Streamlit **App settings → Secrets** (direct HTTPS link to the `.keras` file).\n\n"
            "**Right now:** that path is missing from the deployed repo **and** **`MODEL_URL` is not set** "
            "(or the app was not rebooted after saving Secrets)."
        )
        with st.expander("Option A — commit `artifacts/` into this repo (same folder as your Streamlit main file)"):
            rel_root = ROOT.name or "."
            st.markdown(
                "Your Cloud **Main file path** decides where `artifacts/` must live. "
                "`artifacts/` is always **next to `app.py`**.\n\n"
                f"- If Main file is **`app.py`** (repo root), commit: **`artifacts/{MODEL_FILE}`**\n"
                f"- If Main file is **`med-pred-web/app.py`**, commit: **`med-pred-web/artifacts/{MODEL_FILE}`**\n\n"
                "Putting the model only under `med-pred-web/artifacts/` **does not work** if Streamlit runs **`app.py`** from the **repo root**."
            )
            st.code(
                f"{rel_root}/\n"
                "  app.py          # your Streamlit entrypoint\n"
                "  artifacts/\n"
                f"    {MODEL_FILE}\n"
                f"    {META_FILE}   # optional\n",
                language="text",
            )
        with st.expander("Option B — `MODEL_URL` in Streamlit Cloud"):
            st.markdown(
                "1. Open [share.streamlit.io](https://share.streamlit.io) → your workspace.\n"
                "2. Find this app → **⋮** → **Settings**.\n"
                "3. Open the **Secrets** tab.\n"
                "4. Paste (replace with your real URL):\n\n"
                "```toml\n"
                'MODEL_URL = "https://github.com/<you>/<repo>/releases/download/<tag>/covid_xray_best_model.keras"\n'
                "```\n\n"
                "5. **Save**, then **Manage app → Reboot** (secrets are not always picked up until reboot).\n\n"
                "Use a **Release asset** or other **direct download** link—not the GitHub HTML page for the file."
            )
        st.stop()

    try:
        model = load_model(str(path))
    except Exception as e:
        st.error(
            "TensorFlow could not load the model file. Common causes: corrupt download, "
            "wrong file URL, or train/save TF version very different from Cloud. "
            "Try re-export in Task 9 or fix MODEL_URL."
        )
        with st.expander("Technical details"):
            st.exception(e)
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
