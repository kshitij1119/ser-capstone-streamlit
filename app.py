"""Streamlit SER interface: prediction, notebook metrics, and method summary."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from ser_inference import BundleError, SERPredictor

APP_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = APP_DIR / "model_bundle"
BUNDLE_FILE = BUNDLE_DIR / "bundle.pt"
METADATA_FILE = BUNDLE_DIR / "metadata.json"
RESULTS_FILE = BUNDLE_DIR / "results.json"
CONFUSION_FILE = BUNDLE_DIR / "confusion_matrix.png"

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech Emotion Recognition")
st.caption("RAVDESS capstone · actor-independent speech emotion classification")
st.warning("Predictions estimate acted dataset labels. They do not reveal a person's verified emotional state or diagnose a condition.")

if not BUNDLE_FILE.is_file() or not METADATA_FILE.is_file():
    st.error("The exported SER model bundle is missing.")
    st.markdown("Run the final **Export Streamlit model bundle** cell in the notebook and add `bundle.pt` and `metadata.json` to `model_bundle/` in this repository.")
    st.stop()

@st.cache_resource(show_spinner="Loading classifier heads…")
def load_predictor(bundle: str, metadata: str) -> SERPredictor:
    return SERPredictor(bundle, metadata)

try:
    predictor = load_predictor(str(BUNDLE_FILE), str(METADATA_FILE))
except BundleError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Unable to load model bundle: {exc}")
    st.stop()

predict_tab, results_tab, methods_tab = st.tabs(["Audio demo", "Notebook results", "Method and limitations"])

with predict_tab:
    st.subheader("Classify a speech recording")
    st.caption("Upload a WAV file or record a short sample. Encoders are loaded one at a time to reduce peak memory use.")
    upload_col, record_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("Upload WAV", type=["wav"], key="ser_upload")
    with record_col:
        recorded = st.audio_input("Record speech", key="ser_recording")
    if uploaded is not None and recorded is not None:
        use_recorded = st.radio("Audio source", ["Uploaded file", "Microphone recording"], horizontal=True)
        selected = recorded if use_recorded.startswith("Microphone") else uploaded
    else:
        selected = recorded if recorded is not None else uploaded

    if selected is not None:
        audio_bytes = selected.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Analyze audio", type="primary"):
            try:
                with st.spinner("Processing audio and evaluating the encoder ensemble…"):
                    probabilities, duration, encoder_scores = predictor.predict(audio_bytes)
                best = int(probabilities.argmax())
                st.subheader(f"Prediction: {predictor.labels[best].title()}")
                st.metric("Model score", f"{probabilities[best]:.1%}")
                st.caption(f"Processed duration: {duration:.2f} seconds")
                table = pd.DataFrame({"Emotion": [x.title() for x in predictor.labels], "Score": probabilities})
                table = table.sort_values("Score", ascending=False)
                st.bar_chart(table.set_index("Emotion"), horizontal=True)
                if len(encoder_scores) > 1:
                    with st.expander("Per-encoder ensemble scores"):
                        per_encoder = pd.DataFrame(encoder_scores, index=predictor.labels).T
                        st.dataframe(per_encoder.style.format("{:.1%}"), use_container_width=True)
                st.caption("Scores are relative model outputs and may be miscalibrated outside RAVDESS.")
            except (ValueError, BundleError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with results_tab:
    st.subheader("Results from the executed notebook")
    if RESULTS_FILE.is_file():
        try:
            results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            metrics = {
                "Actor-independent test accuracy": results.get("speaker_independent_test_accuracy (4 unseen actors)"),
                "Notebook test macro-F1": results.get("speaker_independent_test_macro_f1"),
                "Six-fold actor-independent CV accuracy": results.get("speaker_independent_cv_accuracy (6 folds, 24 actors)"),
                "Speaker-dependent reference accuracy": results.get("speaker_dependent_random_split_accuracy (comparison only)"),
            }
            cols = st.columns(len(metrics))
            for col, (label, value) in zip(cols, metrics.items()):
                col.metric(label, f"{float(value):.1%}" if value is not None else "Not reported")
            st.caption("The speaker-dependent random split is a separate reference and is not evidence of new-speaker generalization.")
            if results.get("deployment_head_test_accuracy") is not None:
                left, right = st.columns(2)
                left.metric("Exported app model · unseen-actor accuracy", f"{float(results['deployment_head_test_accuracy']):.1%}")
                right.metric("Exported app model · unseen-actor macro-F1", f"{float(results['deployment_head_test_macro_f1']):.3f}")
            if results.get("cv_fold_mean_accuracy") is not None:
                st.caption(
                    "Cross-validation fold accuracy: "
                    f"{float(results['cv_fold_mean_accuracy']):.1%} ± {float(results['cv_fold_std_accuracy']):.1%}"
                )
            st.write("**Configuration**", {
                "encoders": results.get("encoders"),
                "extra training corpora": results.get("extra_corpora_used"),
                "speaker normalization": results.get("speaker_normalisation"),
            })
            per_class = results.get("per_class", {})
            class_rows = {
                label.title(): {
                    metric: per_class.get(label, {}).get(metric)
                    for metric in ("precision", "recall", "f1-score", "support")
                }
                for label in predictor.labels
                if label in per_class
            }
            if class_rows:
                st.markdown("#### Per-emotion results")
                st.dataframe(pd.DataFrame.from_dict(class_rows, orient="index").style.format(precision=3), use_container_width=True)
            ablation = results.get("ablation", [])
            if ablation:
                st.markdown("#### Validation-selected ablation")
                st.dataframe(pd.DataFrame(ablation), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read exported notebook results: {exc}")
    else:
        st.info("Run the notebook export cell after the evaluation cells to include results.json here.")
    if CONFUSION_FILE.is_file():
        st.image(str(CONFUSION_FILE), caption="Unseen-actor confusion matrix from the notebook")
    else:
        st.info("The notebook confusion matrix can be copied to model_bundle/confusion_matrix.png for display here.")

with methods_tab:
    st.subheader("Project method")
    st.markdown("""
- **Dataset:** RAVDESS speech, with optional RAVDESS song, CREMA-D, TESS, and SAVEE used for training only.
- **Split:** actors, not clips, are assigned to train, validation, and test; six-fold leave-actors-out CV estimates speaker variation.
- **Audio preparation:** mono 16 kHz, silence trim, five-second crop, peak normalization, and per-clip z-score before encoding.
- **Features:** mean and standard deviation pooled from 25 hidden layers of WavLM-large and HuBERT-large.
- **Classifier:** learned softmax weighting over layers followed by a small MLP; ensemble probabilities average seeds and encoders.
- **Training:** augmentations are training-only; the app performs inference and does not train or tune models in Streamlit Cloud.
- **Limits:** acted speech, 24 RAVDESS actors, synthetic/curated recording conditions, and possible domain shift.

The simplified optimized notebook does not implement the first notebook's robustness suite or SHAP/Grad-CAM analyses. Those results are not represented as app features.
""")
    st.caption(f"Loaded {len(predictor.members)} classifier heads across {len(predictor.encoder_ids)} encoder(s).")

