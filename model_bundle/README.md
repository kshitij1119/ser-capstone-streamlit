# Trained model bundle

Run the **Export Streamlit model bundle** cell at the end of the SER notebook, then copy `bundle.pt` and `metadata.json` into this directory.

The bundle contains the trained classifier head and training-only normalization statistics. It should remain small enough for a normal private GitHub repository. The pretrained speech encoder is downloaded from Hugging Face using the `encoder_id` recorded in `metadata.json`.

Commit `bundle.pt` and `metadata.json` here after export so the deployed app can load them. Do not commit dataset files, audio recordings, or credentials.

