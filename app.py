"""Streamlit front end for the RAVDESS speech emotion recognition model."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ser_inference import BundleError, SERPredictor


APP_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = APP_DIR / "model_bundle"
BUNDLE_FILE = BUNDLE_DIR / "bundle.pt"
METADATA_FILE = BUNDLE_DIR / "metadata.json"

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.caption("RAVDESS capstone · actor-independent evaluation · WavLM/HuBERT encoder")
st.info(
    "This model predicts an acted emotion label from speech. It does not determine "
    "a person's true emotional state and is not for diagnosis or high-stakes decisions."
)


@st.cache_resource(show_spinner="Loading the speech encoder and classifier…")
def load_predictor(bundle_file: str, metadata_file: str) -> SERPredictor:
    return SERPredictor(bundle_file, metadata_file)


if not BUNDLE_FILE.is_file() or not METADATA_FILE.is_file():
    st.warning("The trained model bundle is not in this repository yet.")
    st.markdown(
        "Run the **Export Streamlit model bundle** cell at the end of the capstone notebook, "
        "then place `bundle.pt` and `metadata.json` in `ser_streamlit/model_bundle/` and push them to GitHub."
    )
    st.stop()

try:
    predictor = load_predictor(str(BUNDLE_FILE), str(METADATA_FILE))
except BundleError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # show a concise message; deployment details remain in Cloud logs
    st.error(f"Could not load the speech model: {exc}")
    st.stop()

st.caption(
    f"Encoder: `{predictor.encoder_id}` · sample rate: {predictor.sample_rate:,} Hz · "
    f"maximum clip length: {predictor.max_seconds:g} s"
)

audio_file = st.file_uploader("Upload a speech recording", type=["wav"])
if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Classify speech", type="primary"):
        try:
            with st.spinner("Analyzing the recording…"):
                probabilities, duration = predictor.predict(audio_bytes)
            best_index = int(probabilities.argmax())
            st.subheader(f"Predicted label: {predictor.labels[best_index].title()}")
            st.caption(
                f"Model score: {probabilities[best_index]:.1%} · "
                f"processed speech duration: {duration:.2f} s"
            )

            results = pd.DataFrame(
                {"Emotion": [label.title() for label in predictor.labels], "Score": probabilities}
            ).sort_values("Score", ascending=False)
            st.bar_chart(results.set_index("Emotion"), horizontal=True)
            st.caption(
                "Scores are the model's relative class probabilities. They may be "
                "uncalibrated, especially for speakers or recording conditions unlike RAVDESS."
            )
        except (ValueError, BundleError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Audio prediction failed: {exc}")

with st.expander("About this model"):
    st.markdown(
        "- Audio is converted to mono 16 kHz, trimmed, cropped, peak-normalized, and "
        "standardized before encoder inference.\n"
        "- The classifier combines mean and standard-deviation summaries from each "
        "encoder hidden layer.\n"
        "- Evaluation is actor-independent: test actors are excluded from training.\n"
        "- An uploaded recording is processed in memory and is not saved by this app."
    )

