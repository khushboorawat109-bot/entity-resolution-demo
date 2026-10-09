# entity-resolution-demo
Live demo for a business entity resolution model: compare two records, get a match probability and similarity breakdown.

# Entity Resolution Demo

An interactive Streamlit app for the business entity resolution project.
Enter two business records (name, address, country) and the model returns:

- a match / no-match verdict with its probability,
- the normalized text it actually compared,
- a breakdown of the 12 similarity features behind the decision.

You can also override the decision threshold to see how it trades precision
against recall (the model's own threshold is tuned conservatively, because
falsely merging two different businesses is the costly error).

**Main project, with the full pipeline and writeup:** https://github.com/khushboorawat109-bot/Business_entity_resolution

## How it works

Same logic as the full pipeline, in one file (`app.py`):

1. Normalize both records (strip legal suffixes like Inc / Pvt Ltd / SARL,
   expand address abbreviations, remove noise).
2. Compute 12 text similarity features (word and character-trigram overlap,
   sequence similarity, length differences, substring and first-word checks,
   country match).
3. Score the pair with a gradient-boosted classifier (`model.joblib`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The model was trained on business records from a hackathon dataset. The
  data itself is not included here.
- It was trained on US, India and France style records, so results on other
  naming and address conventions may be unreliable.
- `requirements.txt` pins `scikit-learn==1.8.0` on purpose: the saved model
  was trained under that version and may not load under others.
