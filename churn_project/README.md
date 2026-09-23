# Customer Churn Prediction using XGBoost

Predicts whether a telecom customer will leave ("churn") based on their
account and service details, using an XGBoost classifier. Includes full
EDA, preprocessing, model training, evaluation, feature importance
analysis, and a real-time prediction web app (Streamlit + optional Flask API).

---

## ⚠️ Important note about the dataset

This project is designed to run on the **official IBM Telco Customer
Churn dataset**. For your submission, use the real dataset — a synthetic
placeholder is included only so the project runs immediately.

**Option A — direct download, no account needed (easiest):**
IBM hosts the real dataset on GitHub. Run this from your project's `data/` folder:

```bash
# macOS / Linux / VS Code integrated terminal:
curl -o telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# Windows PowerShell:
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -OutFile "telco_churn.csv"
```
This overwrites the synthetic placeholder with the real 7,043-row IBM dataset — same filename, same schema, zero code changes needed.

**Option B — Kaggle mirror (same data, if you prefer Kaggle):**
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Rename it to `telco_churn.csv` and drop it into `data/`, replacing the placeholder

Either way, once the real file is in place, just re-run `python train.py` — no code changes needed.

*(Until you do this, `data/telco_churn.csv` contains a synthetic stand-in with the same 7,043 rows / 21 columns / realistic churn patterns, so you can test-run everything right now.)*

---

## Opening the project in VS Code

1. **Unzip** the downloaded file — you should get a `churn_project` folder.
2. **Open VS Code**, then go to `File > Open Folder...` (Mac: `Cmd+O`, Windows/Linux: `Ctrl+K Ctrl+O`) and select the `churn_project` folder.
   - Or from a terminal, `cd` into the unzipped folder and run:
     ```bash
     code .
     ```
     (this opens VS Code directly in that folder — if `code` isn't recognized, open VS Code, press `Cmd+Shift+P` / `Ctrl+Shift+P`, type "Shell Command: Install 'code' command in PATH", then restart your terminal.)
3. **Open the integrated terminal** inside VS Code: `` Ctrl+` `` (backtick) on Windows/Linux, or `` Cmd+` `` on Mac. All commands below are typed into this terminal.
4. **Select a Python interpreter** (bottom-right corner of VS Code, or `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter") — pick Python 3.9+ if you have multiple versions installed. Install the Microsoft "Python" extension first if VS Code prompts you to.

---

## Project Structure

```
churn_project/
├── data/
│   ├── telco_churn.csv          # dataset (swap in the real IBM file — see above)
│   └── generate_sample_data.py  # regenerates the synthetic placeholder if needed
├── src/
│   ├── preprocessing.py         # cleaning, encoding
│   ├── eda.py                   # exploratory data analysis + plots
│   ├── train.py                 # train + evaluate XGBoost model
│   ├── test_model.py            # sanity-test the saved model on sample customers
│   └── shap_analysis.py         # SHAP explainability (global + per-customer)
├── model/                       # created after training: churn_model.pkl, encoders.pkl, metrics.txt
├── images/                      # all EDA + evaluation plots get saved here
├── app.py                       # Streamlit web app (primary deployment)
├── app_flask.py                 # optional Flask REST API alternative
├── sample_request.json          # example payload for testing the Flask API
└── requirements.txt
```

---

## Step-by-Step: How to Run

### 1. Set up the environment
```bash
cd churn_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the sample dataset
Only needed if you deleted `data/telco_churn.csv` or want a fresh synthetic sample.
Skip this step if you've already dropped in the real IBM CSV.
```bash
cd data
python generate_sample_data.py
cd ..
```

### 3. Run Exploratory Data Analysis
```bash
cd src
python eda.py
```
This prints churn rate / missing-value summary to the console and saves 5 plots
to `images/`: churn distribution, churn by contract type, tenure vs churn,
monthly charges vs churn, and a correlation heatmap.

### 4. Train the model
```bash
python train.py
```
This will:
- Clean and encode the data (`preprocessing.py`)
- Split 80% train / 20% test (stratified)
- Train an `XGBClassifier`
- Print Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the confusion matrix
- Save `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png` to `images/`
- Save the trained model to `model/churn_model.pkl`, encoders to `model/encoders.pkl`,
  and a copy of all metrics to `model/metrics.txt`

Expect **~79–82% accuracy** and **~0.83–0.85 ROC-AUC** on the real IBM dataset
(numbers vary slightly by random seed / dataset version).

### 5. Test the trained model
```bash
python test_model.py
```
Loads the saved model and runs it against three hand-picked sample customers
(a high-risk new customer, a low-risk long-tenure customer, and a borderline
case) so you can eyeball that predictions make directional sense.

### 5.5. Explainability with SHAP (recommended)
```bash
python shap_analysis.py
```
XGBoost's built-in feature importance (from step 4) tells you *which*
features matter most on average, but not *which direction* they push
predictions, or *why* one specific customer was flagged. SHAP fills that
gap. This script saves four plots to `images/`:
- `shap_bar_importance.png` — mean impact of each feature, ranked
- `shap_beeswarm.png` — same ranking, but also shows whether high or low
  values of a feature push predictions toward churn or toward staying
  (e.g. does a *high* Monthly Charges value increase churn risk? this
  plot answers that directly, which the plain importance bar chart can't)
- `shap_waterfall_high_risk_example.png` — a step-by-step breakdown of
  exactly why one specific high-risk customer was predicted to churn
- `shap_waterfall_low_risk_example.png` — same, for a loyal customer

The Streamlit app (step 6 below) also shows a live waterfall explanation
for whichever customer you enter, so you get a "why" alongside every
prediction, not just a probability.

### 6. Launch the web app

**Option A — Streamlit (recommended, interactive UI):**
```bash
cd ..                      # back to project root
streamlit run app.py
```
Opens a browser at `http://localhost:8501` with a form — fill in a customer's
details and click **Predict Churn** to get an instant prediction + probability.

**Option B — Flask REST API:**
```bash
python app_flask.py
```
Runs at `http://127.0.0.1:5000`. Test it with:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```
Expected response:
```json
{"churn_prediction": "Yes", "churn_probability": 0.87}
```

---

## How to Test Everything (quick checklist)

| What to test | Command | What "working" looks like |
|---|---|---|
| Data loads & cleans correctly | `python src/preprocessing.py` | No missing values reported, shape printed |
| EDA runs | `python src/eda.py` | 5 PNGs appear in `images/` |
| Model trains | `python src/train.py` | Accuracy/Precision/Recall/F1 printed, no errors, `model/churn_model.pkl` created |
| Model sanity-check | `python src/test_model.py` | High-risk customer → CHURN, low-risk customer → NO CHURN |
| SHAP explainability | `python src/shap_analysis.py` | 4 PNGs appear in `images/` (bar, beeswarm, 2 waterfalls) |
| Streamlit app | `streamlit run app.py` | Form loads, prediction + probability bar shown on click, SHAP waterfall shown below it |
| Flask API | `python app_flask.py` + curl command above | JSON response with prediction + probability |

---

## Technology Stack
- **Language:** Python 3.9+
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** Scikit-learn (preprocessing, splitting, metrics), XGBoost (model)
- **Explainability:** SHAP (global + per-customer churn driver analysis)
- **Deployment:** Streamlit (web UI) / Flask (REST API)
- **Version control:** GitHub (see below)

---

## Putting this on GitHub (recommended for submission)
```bash
git init
git add .
echo "venv/
__pycache__/
*.pyc
model/*.pkl" > .gitignore
git add .gitignore
git commit -m "Customer Churn Prediction using XGBoost"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```
> Tip: `.gitignore` excludes the trained `.pkl` files since they're
> regenerated by `train.py` — this keeps the repo small and shows graders
> you understand reproducible ML workflows rather than just committing a
> black-box model file.

---

## Making this stand out (tips for a graded/selected submission)
1. **Swap in the real IBM dataset** (see note at the top) — a reviewer
   familiar with this dataset will notice a synthetic one immediately.
2. **Add the metrics + plots to your report/slides** — `model/metrics.txt`
   and the `images/` folder give you everything needed for a results section.
3. **Push to GitHub with a clean README** (this file already doubles as one)
   and a couple of commits showing iteration, not one giant commit.
4. **Mention class imbalance handling** — the real dataset is ~73%/27%
   No/Yes; `train.py` already uses `scale_pos_weight` to address this,
   which is worth calling out in your write-up as it shows you understand
   why plain accuracy alone is a misleading metric for imbalanced problems.
5. **Record a 60-90 second demo video** of the Streamlit app for your
   submission — a working demo consistently stands out more than a
   notebook full of static numbers.
6. **Optional stretch goals** if you have time: hyperparameter tuning
   with `GridSearchCV`/`Optuna`, or a Dockerfile for one-command deployment.

---

## Expected Outcome
Given a customer's account details, the system predicts churn risk and
returns a probability score, helping a business prioritize retention
efforts (discounts, contract upgrades, proactive support outreach) toward
the customers most likely to leave.
