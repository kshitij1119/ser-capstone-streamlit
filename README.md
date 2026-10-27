# Speech Emotion Recognition — Streamlit app

A Streamlit interface for the RAVDESS speech-emotion capstone. It provides single-recording inference with WAV upload or microphone input, averages the notebook's encoder/seed classifier heads, and displays evaluation outputs exported from the notebook.

## Project files

- `app.py` — Streamlit interface, prediction, evaluation results, and method summary.
- `ser_inference.py` — WAV preprocessing, encoder feature extraction, and ensemble inference.
- `export_bundle.py` — helper for serializing fitted notebook heads and their feature statistics.
- `requirements.txt` — Streamlit Cloud runtime dependencies.
- `model_bundle/` — receives `bundle.pt`, `metadata.json`, `results.json`, and `confusion_matrix.png` from the notebook export.

## Generate the model bundle

1. Open and run the notebook `SER_capstone_optimized_explained.ipynb` through its final **Export Streamlit model bundle** cell. Run it after the notebook's data preparation, feature extraction, split, training, and evaluation cells have completed.
2. The export cell trains one classifier head for every configured encoder and seed, using the actor-independent train/validation/test split and no speaker normalization for single-utterance inference.
3. Copy all four generated files—`bundle.pt`, `metadata.json`, `results.json`, and `confusion_matrix.png`—into this repository's `model_bundle/` directory and commit them. The encoder weights are downloaded from Hugging Face at runtime; do not include dataset audio or credentials.

The app loads one large pretrained encoder at a time. The classifier bundle contains the fitted heads and training feature statistics. A compatible Hugging Face model must be publicly downloadable or accessible to the deployment runtime.

## Deploy to Streamlit Community Cloud

1. Create a Streamlit Community Cloud app from this repository.
2. Select branch `main` and set the app entry point to `app.py` at the repository root.
3. Deploy and inspect the app logs during the first launch as Transformers downloads the configured encoders.

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
streamlit run app.py
```

## What the app represents

- Audio predictions use the exported encoder/seed head ensemble and the notebook's single-clip preprocessing.
- The results tab displays saved notebook metrics, per-class results, ablation output, and the unseen-actor confusion matrix when those exports are present.
- Training, cross-validation, and ablation are run in the notebook; the Streamlit app presents saved outputs and performs inference only.
- The optimized notebook does not implement the earlier notebook's robustness suite, SHAP, or Grad-CAM analyses. This deployment does not claim those analyses.
- Report actor-independent test/CV results separately from the speaker-dependent random-split reference. RAVDESS contains acted emotions; predictions are model outputs, not verified emotional states.

