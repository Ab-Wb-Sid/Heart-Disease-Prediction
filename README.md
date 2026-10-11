# Heart Disease Prediction System

This repository contains an AIC354 Assignment 2 project for predicting heart disease presence from the UCI Heart Disease dataset.

The project uses exactly four input features:

- `age`
- `cp`
- `thalach`
- `oldpeak`

The target is created from the UCI `num` column:

- `0` means no disease.
- Values greater than `0` mean disease present.

## Project Files

- `heart_disease_prediction.ipynb` - complete notebook with data inspection, cleaning, model training, evaluation, saving, and reload test.
- `data/heart_disease_cleveland_selected.csv` - cleaned selected dataset created from the UCI processed Cleveland file.
- `models/heart_disease_pipeline.joblib` - saved fitted scikit-learn pipeline.
- `models/model_metadata.json` - feature ranges, class distribution, and evaluation metrics used by the app.
- `app.py` - Streamlit app that loads the saved model and predicts from exactly four inputs.
- `requirements.txt` - deployment dependencies.
- `.gitignore` - excludes virtual environments, caches, secrets, and logs.
- `heart+disease/` - downloaded UCI Heart Disease dataset folder.

## Dataset

Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

This project uses `heart+disease/processed.cleveland.data` because the UCI documentation describes it as the processed Cleveland dataset commonly used in machine-learning experiments. The raw processed file has 303 rows and 14 columns. The selected four input columns contain no missing values in this variant.

## Model

The notebook trains a scikit-learn `Pipeline`:

- median imputation and standard scaling for numeric features: `age`, `thalach`, `oldpeak`
- most-frequent imputation and one-hot encoding for categorical feature: `cp`
- logistic regression classifier

Held-out test results from the notebook:

- Accuracy: 0.852
- Precision: 0.913
- Recall: 0.750
- F1-score: 0.824
- Confusion matrix: `[[31, 2], [7, 21]]`

These metrics come from the executed notebook using a reproducible random seed.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

If the model file is missing, run `heart_disease_prediction.ipynb` first.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose this repository: `Ab-Wb-Sid/Heart-Disease-Prediction`.
4. Select the deployment branch, usually `main`.
5. Set the main file path to `app.py`.
6. Deploy and copy the public app URL.
7. Paste the live URL into the first Markdown cell of `heart_disease_prediction.ipynb`.

## Disclaimer

This is an educational classroom project only. It is not a medical diagnosis tool and must not be used for clinical or health decisions.
