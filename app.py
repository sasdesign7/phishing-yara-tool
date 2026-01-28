import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Threat Analysis Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Threat Analysis Dashboard")

tool = st.sidebar.radio(
    "Select Analysis Module",
    [
        "🚨 Phishing URL Detection",
        "🛡️ YARA Rule Recommendation"
    ]
)

# ==================================================
# 🚨 PHISHING URL DETECTOR
# ==================================================
if tool == "🚨 Phishing URL Detection":

    st.header("🚨 Phishing URL Detection")
    st.caption("Machine-learning based phishing detection using URL structure.")

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    def url_features(u):
        p = urlparse(u)
        return {
            "url_len": len(u),
            "dot_count": u.count("."),
            "slash_count": u.count("/"),
            "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", u))),
            "has_https": int(u.startswith("https")),
            "has_at": int("@" in u),
            "has_dash": int("-" in u),
            "query_len": len(p.query),
            "path_depth": len([x for x in p.path.split("/") if x]),
            "special_chars": len(re.findall(r"[^a-zA-Z0-9]", u))
        }

    # ---------------------------
    # LOAD DATASET
    # ---------------------------
    data = pd.read_csv("processed_urls.csv")
    data.columns = data.columns.str.lower()

    data["label"] = data["label"].replace(
        {"phishing": 1, "legitimate": 0, -1: 1}
    ).astype(int)

    X = pd.DataFrame([url_features(u) for u in data["url"].astype(str)])
    y = data["label"]

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    model = GradientBoostingClassifier()
    model.fit(Xtr, ytr)

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_url = st.text_input("🔗 Enter URL")

    if st.button("Analyze"):
        if not user_url.strip():
            st.warning("Please enter a valid URL.")
        else:
            feats = pd.DataFrame([url_features(user_url)])
            prob = model.predict_proba(feats)[0][1]

            if prob > 0.7:
                st.error(f"High Risk Phishing URL — Confidence: {prob:.2f}")
            elif prob > 0.4:
                st.warning(f"Suspicious URL — Confidence: {prob:.2f}")
            else:
                st.success(f"Likely Legitimate — Confidence: {prob:.2f}")

            st.subheader("Extracted Features")
            st.json(feats.to_dict(orient="records")[0])

# ==================================================
# 🛡️ YARA RULE RECOMMENDER
# ==================================================
if tool == "🛡️ YARA Rule Recommendation":

    st.header("🛡️ YARA Rule Recommendation")
    st.caption("TF-IDF similarity matching against known YARA rules.")

    # ---------------------------
    # LOAD YARA DATA
    # ---------------------------
    @st.cache_data
    def load_yara():
        df = pd.read_csv("yara_rules_clean.csv")
        return df.dropna(subset=["rule_text"])

    yara_df = load_yara()

    # ---------------------------
    # STRING EXTRACTION
    # ---------------------------
    def extract_strings(rule):
        matches = re.findall(r'\$[a-zA-Z0-9_]+\s*=\s*(.+)', rule)
        return " ".join(matches).lower()

    yara_df["strings"] = yara_df["rule_text"].apply(extract_strings)
    yara_df = yara_df[yara_df["strings"].str.len() > 10]

    # ---------------------------
    # TF-IDF MODEL
    # ---------------------------
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        max_features=40000
    )

    tfidf_matrix = vectorizer.fit_transform(yara_df["strings"])

    # ---------------------------
    # USER INPUT
    # ---------------------------
    indicators = st.text_area(
        "Paste malware strings / indicators",
        height=180
    )

    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("Find Matching Rules"):
        if not indicators.strip():
            st.warning("Please provide malware indicators.")
        else:
            q_vec = vectorizer.transform([indicators.lower()])
            sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

            yara_df["score"] = sims
            results = yara_df.sort_values(
                "score", ascending=False
            ).head(top_k)

            st.subheader("Recommended YARA Rules")

            for _, row in results.iterrows():
                with st.expander(
                    f"{row.get('rule_name', 'Unnamed Rule')} "
                    f"(Score: {row['score']:.2f})"
                ):
                    st.code(row["rule_text"], language="yara")
