# Trained model bundle

Run the final **Export Streamlit model bundle** cell in `SER_capstone_optimized_explained.ipynb`. Copy these generated files into this directory:

- `bundle.pt` — classifier heads and training-only feature normalization statistics.
- `metadata.json` — labels, encoder IDs, and preprocessing settings.
- `results.json` — saved evaluation metrics and per-class results.
- `confusion_matrix.png` — unseen-actor confusion matrix.

The pretrained encoders are loaded from Hugging Face using the IDs in the metadata. The Git ignore rules allow these export files to be committed; do not commit dataset audio or credentials.

