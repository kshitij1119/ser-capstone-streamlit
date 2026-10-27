# SER Streamlit deployment

This app serves the RAVDESS speech-emotion classifier from the capstone notebook. It expects a small trained head bundle in `model_bundle/`; the WavLM/HuBERT encoder is loaded from its Hugging Face model ID at app startup.

## Files

- `app.py` — Streamlit interface for WAV upload and eight-class probability display.
- `ser_inference.py` — audio preprocessing, encoder inference, and classifier loading.
- `export_bundle.py` — serializes a trained notebook head and its feature normalization.
- `requirements.txt` — app dependencies for Streamlit Community Cloud.
- `model_bundle/` — add `bundle.pt` and `metadata.json` after exporting from the notebook.

## Create the trained bundle

Run the final **Export Streamlit model bundle** cell in `SER_capstone_optimized_explained.ipynb` after the data, embeddings, split, and training functions have been created. It trains a deployable head without speaker normalization (required for a single uploaded utterance), evaluates that head on the held-out actor test split, and writes `model_bundle/bundle.pt` plus `metadata.json`.

Copy both generated files into this app's `model_bundle/` directory. The classifier bundle is small; the pretrained encoder weights are downloaded by Transformers when the app starts.

## Deploy on Streamlit Community Cloud

1. Push this directory's files to the private `kshitij1119/ser-capstone-streamlit` GitHub repository.
2. Add the two generated bundle files under `model_bundle/` and push them too.
3. In Streamlit Community Cloud, create an app from that repository, select branch `main`, and set the app file to `app.py`.
4. Use Python 3.11 or 3.12 in advanced settings if dependency resolution for the default runtime fails.
5. Check Cloud logs after the first launch. The first launch downloads the large encoder; memory or startup constraints may require a smaller encoder and a newly trained matching head.

## Local run

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
streamlit run app.py
```

## Evaluation and privacy notes

- The uploaded WAV is processed in memory; the app does not save it.
- Predictions represent the model's output for acted emotion labels, not a person's verified emotional state.
- Actor-independent results must be reported separately from any speaker-dependent random-split comparison.
- If the notebook selected speaker normalization, this deployment head is trained again with that option disabled so one-clip inference matches its training preprocessing.

