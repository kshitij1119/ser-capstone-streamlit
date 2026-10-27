# Trained model bundle

Run the **Export Streamlit model bundle** cell at the end of the SER notebook, then copy `bundle.pt` and `metadata.json` into this directory.

The bundle contains the trained classifier head and training-only normalization statistics. The pretrained speech encoder is downloaded from Hugging Face using the `encoder_id` recorded in `metadata.json`.

Model weights are ignored by Git until they are explicitly added. Do not commit dataset files, audio recordings, or credentials.

