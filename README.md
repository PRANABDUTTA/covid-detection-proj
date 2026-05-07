# Chest X-ray classification (Streamlit)

COVID-19 vs Normal vs Viral Pneumonia — inference UI for the model exported from **`Covid19_Chest_Xrays_CNN.ipynb`** (Task 9).

**Repo folder:** `projects/08-cnn/med-pred-web`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place exports next to `app.py`:

- `artifacts/covid_xray_best_model.keras` (required)
- `artifacts/covid_xray_best_model_metadata.json` (optional; class order + metrics)

Then:

```bash
streamlit run app.py
```

## Deploy (GitHub + Streamlit Community Cloud)

See **[DEPLOY.md](DEPLOY.md)** — same workflow as `telco-churn-streamlit`: push this repo, deploy with **Main file:** `app.py`, Python **3.11** or **3.12**.

## Artifacts

Train/export in Colab or local Jupyter using Task 9 in the CNN notebook, download **`covid_xray_best_model.keras`** (and JSON if generated), copy into **`artifacts/`**.

If the model file is large (>100 MB), use **[Git LFS](https://git-lfs.com)** or adjust `.gitignore` and host the file elsewhere; see DEPLOY.md.
