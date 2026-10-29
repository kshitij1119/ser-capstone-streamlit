# Speech Emotion Recognition Capstone

A Streamlit demonstration for an audio-only speech emotion recognition (SER) capstone. The app accepts a WAV file or microphone recording, runs the exported six-head ensemble, and displays an emotion estimate with relative class scores and evaluation outputs.

> The model predicts labels from acted speech datasets. Its output does not verify a person's emotional state or diagnose a condition.

## How the project works

1. **Data:** RAVDESS speech is the main training and evaluation corpus. CREMA-D and TESS are used as training-only auxiliary data.
2. **Speaker-independent split:** RAVDESS actors are separated into train, validation, and test groups. The standard held-out split uses 16 training actors, 4 validation actors, and 4 unseen test actors (240 test clips). CREMA-D and TESS are never used for validation or testing.
3. **Audio preparation:** recordings are converted to mono at 16 kHz, silence-trimmed, cropped to at most 5 seconds, peak-normalized, padded when short, and standardized before encoder inference. Training augmentation includes noise, pitch shifting, and time stretching.
4. **Feature extraction:** frozen WavLM-Large and HuBERT-Large encoders provide hidden states. Each of the 25 hidden states is summarized by temporal mean and standard deviation, producing a 2,048-value vector per layer.
5. **Classification:** each encoder has its own learned layer-weighted MLP head. Three random-seed heads are trained per encoder; their six class-probability vectors are averaged. Encoder features are not concatenated.
6. **Inference:** the Streamlit app applies the same single-recording preprocessing, loads one pretrained encoder at a time, evaluates the saved heads, and displays the averaged scores.

The eight output classes are **neutral, calm, happy, sad, angry, fearful, disgust, and surprised**.

## Current exported results

The model bundle's saved actor-independent results report:

| Evaluation | Result |
|---|---:|
| RAVDESS unseen-actor test accuracy | **80.42%** |
| RAVDESS unseen-actor test macro-F1 | **0.8036** |
| Test set | **240 clips from 4 actors** |

Macro-F1 is the unweighted average of the per-class F1 scores. The app's Notebook results tab also shows per-class metrics and the confusion matrix.

- [Detailed metrics](model_bundle/results.json)
- [Unseen-actor confusion matrix](model_bundle/confusion_matrix.png)

These are the results stored with the current exported bundle. Report them separately from any older notebook run or speaker-dependent random split.

## Repository files

| File | Purpose |
|---|---|
| app.py | Streamlit interface, audio input, prediction display, results, and method notes |
| ser_inference.py | WAV decoding, preprocessing, encoder feature extraction, and ensemble inference |
| export_bundle.py | Helper for serializing trained classifier heads and feature statistics |
| requirements.txt | Python dependencies used by Streamlit Community Cloud |
| model_bundle/bundle.pt | Six classifier heads and their training feature statistics |
| model_bundle/metadata.json | Labels, encoder identifiers, audio settings, and feature dimensions |
| model_bundle/results.json | Exported evaluation metrics and per-class results |
| model_bundle/confusion_matrix.png | Unseen-actor confusion matrix |

The model bundle is already included in this repository. The pretrained encoder weights are downloaded from Hugging Face when the app runs; dataset audio is not included.

## Generate or replace the model bundle

To create a new bundle after changing the training data or model:

1. Open **SER_capstone_optimized_explained.ipynb** in Google Colab.
2. Run the notebook cells in order through data preparation, actor splitting, feature extraction, classifier training, and evaluation.
3. Run the final **Export Streamlit model bundle** cell.
4. Copy the generated bundle files into this repository's **model_bundle/** directory and commit the update:
   - bundle.pt
   - metadata.json
   - results.json
   - confusion_matrix.png

The final app bundle must use single-utterance inference without speaker normalization. Do not add dataset audio or credentials to this repository.

## Deploy to Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
2. Select the GitHub repository **kshitij1119/ser-capstone-streamlit**.
3. Choose the **main** branch and set the app entry point to **app.py** at the repository root.
4. Deploy. The root **requirements.txt** declares the Python dependencies.
5. Check the deployment logs on the first start. The runtime installs dependencies and downloads the public WavLM-Large and HuBERT-Large encoders from Hugging Face. The first launch may take longer while these downloads complete.

Community Cloud currently defaults to Python 3.12; the Python version can be selected in the app's advanced settings during creation. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). Updates committed to the connected GitHub repository can trigger an app redeployment.

## Run locally

~~~bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
streamlit run app.py
~~~

## What the app does and does not do

- Provides single-recording prediction from a WAV upload or microphone input.
- Displays the predicted class, relative class scores, exported metrics, and confusion matrix.
- Performs inference only; training and evaluation are run in the notebook.
- Uses acted speech datasets. Results do not establish natural-conversation or cross-corpus generalization.
- Does not include the earlier notebook's robustness suite, SHAP, or Grad-CAM analyses.
- Displays model scores, which may be miscalibrated and are not measurements of a person's internal emotional state.

## References

1. S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),” *PLOS ONE*, vol. 13, no. 5, e0196391, 2018. [doi:10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391).
2. H. Cao et al., “CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset,” *IEEE Transactions on Affective Computing*, vol. 5, no. 4, pp. 377-390, 2014. [doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244).
3. K. Pichora-Fuller and K. Dupuis, “Toronto emotional speech set (TESS),” Borealis, 2020. [doi:10.5683/SP2/E8H2MF](https://doi.org/10.5683/SP2/E8H2MF).
4. S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505-1518, 2022. [doi:10.1109/JSTSP.2022.3188113](https://doi.org/10.1109/JSTSP.2022.3188113).
5. W.-N. Hsu et al., “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3451-3460, 2021. [doi:10.1109/TASLP.2021.3122291](https://doi.org/10.1109/TASLP.2021.3122291).
6. Streamlit Community Cloud, [deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
