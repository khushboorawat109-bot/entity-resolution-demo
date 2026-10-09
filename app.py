"""
Business Entity Resolution — Interactive Demo
Enter two business records and see whether the trained model thinks they're
the same real-world business, plus a breakdown of every similarity feature
behind that decision.

Self-contained: no imports from the original project's src/ folder, so this
runs standalone on Streamlit Community Cloud / Hugging Face Spaces with just
`app.py`, `model.joblib`, and `requirements.txt`.
"""
import difflib
import re
import unicodedata

import joblib
import numpy as np
import streamlit as st

# ----------------------------------------------------------------------
# Normalization (same logic as the original pipeline's normalize.py)
# ----------------------------------------------------------------------
LEGAL_FORM_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "l l c", "ltd", "limited", "llp", "l l p", "pllc", "pc", "plc",
    "pvt", "private", "pvtltd",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "fils", "dba",
}
_LEGAL_FORM_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(t) for t in LEGAL_FORM_TOKENS),
                              key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "sq": "square", "hwy": "highway",
    "pkwy": "parkway", "apt": "apartment", "bldg": "building",
    "fl": "floor", "ste": "suite", "no": "number", "hno": "house number",
}
_WS_RE = re.compile(r"\s+")
_LEADING_NOISE_RE = re.compile(r"^[\-\#\<\>\|\*\.\s]+")
_PUNCT_RE = re.compile(r"[^\w\s&]", flags=re.UNICODE)
_URL_SUFFIX_RE = re.compile(r"\.(com|in|co|org|net|fr)\b", flags=re.IGNORECASE)
_TRAILING_URL_RE = re.compile(r"\|\s*www\.\S+$")


def _dedupe_consecutive(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def normalize_name(raw: str) -> str:
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw))
    s = _TRAILING_URL_RE.sub("", s)
    s = _LEADING_NOISE_RE.sub("", s)
    s = s.replace("&", " and ")
    s = _URL_SUFFIX_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _LEGAL_FORM_PATTERN.sub(" ", s)
    tokens = [t for t in _WS_RE.split(s.strip()) if t]
    return " ".join(_dedupe_consecutive(tokens))


def normalize_address(raw: str) -> str:
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw))
    s = _LEADING_NOISE_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in _WS_RE.split(s.strip()) if t]
    return " ".join(_dedupe_consecutive(tokens))


def normalize_country(raw: str) -> str:
    return str(raw).strip().lower() if raw else ""


# ----------------------------------------------------------------------
# Feature engineering (same logic as the original pipeline's features.py)
# ----------------------------------------------------------------------
def _char_ngrams(s, n=3):
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _token_set(s):
    return set(s.split()) if s else set()


def _seq_ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()


FEATURE_NAMES = [
    "name_tok_jaccard", "addr_tok_jaccard",
    "name_char_jaccard", "addr_char_jaccard",
    "name_seq_ratio", "addr_seq_ratio",
    "name_len_diff", "addr_len_diff",
    "name_substr", "first_tok_match",
    "country_match",
    "blend_tok_jaccard",
]

FEATURE_LABELS = {
    "name_tok_jaccard": "Name word overlap",
    "addr_tok_jaccard": "Address word overlap",
    "name_char_jaccard": "Name character similarity",
    "addr_char_jaccard": "Address character similarity",
    "name_seq_ratio": "Name sequence similarity",
    "addr_seq_ratio": "Address sequence similarity",
    "name_len_diff": "Name length difference (lower = more similar)",
    "addr_len_diff": "Address length difference (lower = more similar)",
    "name_substr": "One name contains the other",
    "first_tok_match": "First word of name matches",
    "country_match": "Country matches",
    "blend_tok_jaccard": "Combined name+address word overlap",
}


def pair_features(name_a, addr_a, name_b, addr_b, country_a, country_b):
    na_tok, nb_tok = _token_set(name_a), _token_set(name_b)
    aa_tok, ab_tok = _token_set(addr_a), _token_set(addr_b)

    name_tok_jac = _jaccard(na_tok, nb_tok)
    addr_tok_jac = _jaccard(aa_tok, ab_tok)
    name_char_jac = _jaccard(_char_ngrams(name_a), _char_ngrams(name_b))
    addr_char_jac = _jaccard(_char_ngrams(addr_a), _char_ngrams(addr_b))
    name_seq = _seq_ratio(name_a, name_b)
    addr_seq = _seq_ratio(addr_a, addr_b)
    name_len_diff = abs(len(name_a) - len(name_b)) / max(len(name_a), len(name_b), 1)
    addr_len_diff = abs(len(addr_a) - len(addr_b)) / max(len(addr_a), len(addr_b), 1)
    name_substr = float(bool(name_a) and bool(name_b) and (name_a in name_b or name_b in name_a))
    first_tok_match = float(bool(na_tok) and bool(nb_tok) and
                             (name_a.split()[0] == name_b.split()[0]))
    country_match = float(country_a == country_b and country_a != "")
    blend_tok_jac = _jaccard(na_tok | aa_tok, nb_tok | ab_tok)

    return [name_tok_jac, addr_tok_jac, name_char_jac, addr_char_jac,
            name_seq, addr_seq, name_len_diff, addr_len_diff,
            name_substr, first_tok_match, country_match, blend_tok_jac]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["threshold"]


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Business Entity Resolution Demo", page_icon="🔗", layout="centered")

st.title("🔗 Business Entity Resolution")
st.caption(
    "Enter two business records below. The model — trained on noisy, "
    "multi-source business data (typos, abbreviations, inconsistent "
    "addresses) — predicts whether they refer to the same real-world "
    "business, using the same similarity features and classifier from the "
    "full-scale pipeline."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Record A")
    name_a = st.text_input("Business name", "Acme Corp", key="name_a")
    addr_a = st.text_input("Address", "123 Main St, Springfield, IL", key="addr_a")
    country_a = st.text_input("Country", "US", key="country_a")
with col2:
    st.subheader("Record B")
    name_b = st.text_input("Business name", "Acme Corporation", key="name_b")
    addr_b = st.text_input("Address", "123 Main Street, Springfield, Illinois", key="addr_b")
    country_b = st.text_input("Country", "US", key="country_b")

threshold_override = st.slider(
    "Decision threshold (higher = more conservative / precision-favoring)",
    min_value=0.05, max_value=0.95, value=None, step=0.05,
    help="Defaults to the model's own tuned threshold if left untouched.",
)

if st.button("Compare records", type="primary", use_container_width=True):
    model, tuned_threshold = load_model()
    threshold = threshold_override if threshold_override is not None else tuned_threshold

    nn_a, na_a, nc_a = normalize_name(name_a), normalize_address(addr_a), normalize_country(country_a)
    nn_b, na_b, nc_b = normalize_name(name_b), normalize_address(addr_b), normalize_country(country_b)

    feats = pair_features(nn_a, na_a, nn_b, na_b, nc_a, nc_b)
    X = np.asarray([feats], dtype=np.float32)
    prob = model.predict_proba(X)[0, 1]
    is_match = prob >= threshold

    st.divider()
    if is_match:
        st.success(f"### ✅ Likely the SAME business  \nMatch probability: **{prob:.1%}**  (threshold: {threshold:.0%})")
    else:
        st.error(f"### ❌ Likely DIFFERENT businesses  \nMatch probability: **{prob:.1%}**  (threshold: {threshold:.0%})")

    with st.expander("Show normalized text (what the model actually compared)"):
        st.write(f"**Record A** — name: `{nn_a}` | address: `{na_a}` | country: `{nc_a}`")
        st.write(f"**Record B** — name: `{nn_b}` | address: `{na_b}` | country: `{nc_b}`")

    st.subheader("Feature breakdown")
    st.caption("Each similarity signal the model used to make its decision.")
    for fname, fval in zip(FEATURE_NAMES, feats):
        label = FEATURE_LABELS[fname]
        display_val = 1 - fval if "len_diff" in fname else fval  # invert diffs so bar reads "more similar = fuller"
        st.progress(min(max(display_val, 0.0), 1.0), text=f"{label}: {fval:.2f}")

st.divider()
st.caption(
    "Built for a business entity resolution challenge — matching records across noisy, "
    "multi-source business data at scale (1.7M+ records). "
    "[Read the full writeup on GitHub](#)."
)
