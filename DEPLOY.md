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
4. **Before you deploy:** open **Advanced settings** and set **Python version** to **3.12** (or **3.11**).  
   **Do not use Python 3.14** for this app: TensorFlow has **no matching wheels** for 3.14 on Linux, so you will see errors like *“no wheels with a matching Python ABI tag”* or *“No matching distribution found for tensorflow”*.
5. Click **Deploy**.

Cloud installs from **`requirements.txt`** and runs **`streamlit run app.py`**.

### If the app was already created on Python 3.14

The UI may default to a new Python; fix it without guessing:

1. Open your app on Community Cloud → **Manage app** (⋮) → **Settings**.
2. Under **Python version**, choose **3.12.10** (or any **3.12.x** / **3.11.x** offered).
3. **Save** and let the app **rebuild**, or use **Reboot** after saving.

If you **cannot** change Python on the existing deployment, **delete** the app and **deploy again**, making sure **Advanced settings → Python 3.12** is set **before** the first build.

Official reference: [Upgrade your app’s Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python-version).

## 4. After model updates

Re-export from the notebook (Task 9), replace files under **`artifacts/`**, commit, push — Cloud redeploys (or **Manage app → Reboot**).

## Troubleshooting

- **Python 3.14 + `tensorflow` / “unsatisfiable” / “no matching distribution”**  
  TensorFlow does not support Python **3.14** on Streamlit’s Linux build image yet. **Set Python to 3.12** in app settings (see section 3) and redeploy. This is the most common cause of a failed **uv** / **pip** install for this repo.

- **`Missing covid_xray_best_model.keras`:** Ensure the `.keras` file is committed (or on LFS) and path is **`artifacts/covid_xray_best_model.keras`** (case-sensitive on Linux).

- **Slow / failing pip on Cloud:** After Python is 3.12, first install can still take several minutes (TensorFlow is large).

- **TensorFlow import / load errors at runtime:** Train and save the model on a **similar** TensorFlow 2.x version when possible, or re-export in Colab after upgrading TF.
