# Deploy chest X-ray app (GitHub + Streamlit Community Cloud)

Mirror of the **telco-churn-streamlit** workflow: standalone folder → GitHub → [Streamlit Cloud](https://share.streamlit.io).

## 1. Prepare artifacts

From **`Covid19_Chest_Xrays_CNN.ipynb`**, run **Task 9** after training. That saves:

- `covid_xray_best_model.keras`
- `covid_xray_best_model_metadata.json` (optional)

Copy both into **`med-pred-web/artifacts/`** (same names).

**Large models:** GitHub blocks files **> 100 MB**. Options:

- **[Git LFS](https://git-lfs.com)** to track `artifacts/*.keras`, or  
- Remove `artifacts/*.keras` from `.gitignore` only after switching to LFS, or  
- Store the file externally and download at startup (not implemented in the default app).

## 2. Initialize Git and push (if not already)

From **`med-pred-web`**:

```bash
git init
git add .
git commit -m "Add chest X-ray Streamlit app"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Create an **empty** GitHub repo first (no README) if this is the first push.

## 3. Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → select repo + branch **`main`**.
3. **Main file path:** `app.py`
4. **Advanced settings → Python version:** **3.11** or **3.12** (recommended). Avoid bleeding-edge Python if TensorFlow wheels lag.
5. Deploy.

Cloud installs from **`requirements.txt`** and runs **`streamlit run app.py`**.

## 4. After model updates

Re-export from the notebook (Task 9), replace files under **`artifacts/`**, commit, push — Cloud redeploys (or **Manage app → Reboot**).

## Troubleshooting

- **`Missing covid_xray_best_model.keras`:** Ensure the `.keras` file is committed (or on LFS) and path is **`artifacts/covid_xray_best_model.keras`** (case-sensitive on Linux).
- **Slow / failing pip on Cloud:** Try pinning TensorFlow in `requirements.txt` as already done; use Python **3.12** not experimental versions if builds stall.
- **TensorFlow import errors:** Match TensorFlow version roughly to the environment where you saved the model when possible.
