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

**Critical:** Your **build logs must show** something like `Python 3.12.x` (or `3.11.x`). If they show **`Python 3.14.x`**, TensorFlow **will not install** on Linux (no wheels yet). That failure is **not** fixed by editing `requirements.txt` alone.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app** (or equivalent).
3. Select repo + branch **`main`**, **Main file path:** `app.py` (or the path to your entrypoint in that repo).
4. **Before you click Deploy:** open **Advanced settings** (sometimes labeled **▼** or **Optional configuration**).
5. In **Python version**, pick **3.12.x** (or **3.11.x**). **Do not** leave **3.14** selected.
6. Click **Save** on the Advanced modal if present, then **Deploy**.

Cloud installs from **`requirements.txt`** and runs **`streamlit run app.py`**.

### If logs still show Python 3.14 after you “fixed” settings

On Community Cloud, **Python is chosen when the app is created** (see [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): **Advanced settings** → **Python version**). The [App settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings) page documents URL, sharing, and secrets — **not** changing the interpreter. So if your deployment is stuck on **3.14**:

1. **[Delete your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/delete-your-app)** from the workspace (⋮ → delete, or the dashboard delete flow).
2. **Create app** again from the same GitHub repo.
3. On the **first** deploy dialog, open **Advanced settings** and select **Python 3.12** (or **3.11**) **before** the first build completes.
4. Confirm in logs: `Using Python 3.12` / `3.11` — **not** `3.14`.

## 4. After model updates

Re-export from the notebook (Task 9), replace files under **`artifacts/`**, commit, push — Cloud redeploys (or **Manage app → Reboot**).

## Troubleshooting

- **Python 3.14 + `tensorflow` / “unsatisfiable” / “no matching distribution”**  
  TensorFlow has **no wheels for Python 3.14** on Streamlit’s Linux image. You cannot fix this from `requirements.txt` alone. **Delete the app** on Community Cloud and **create it again**; in **Advanced settings** pick **Python 3.12** or **3.11** before the first deploy (see section 3). Confirm logs show `Python 3.12` / `3.11`, not `3.14`.

- **`Missing covid_xray_best_model.keras`:** Ensure the `.keras` file is committed (or on LFS) and path is **`artifacts/covid_xray_best_model.keras`** (case-sensitive on Linux).

- **Slow / failing pip on Cloud:** After Python is 3.12, first install can still take several minutes (TensorFlow is large).

- **TensorFlow import / load errors at runtime:** Train and save the model on a **similar** TensorFlow 2.x version when possible, or re-export in Colab after upgrading TF.
