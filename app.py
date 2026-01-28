import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cyber Defense Utilities",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Cyber Defense Utilities")

module = st.sidebar.selectbox(
    "Choose Module",
    ["Phishing Link Analyzer", "YARA Rule Matcher"]
)

# ==================================================
# PHISHING LINK ANALYZER
# ==================================================
if module == "Phishing Link Analyzer":

    st.header("🚨 Phishing Link Analyzer")
    st.write("Machine learning based detection using URL structure.")

    # ---------------------------
    # FEATURE EXTRACTION
    # ---------------------------
    def extract_url_metrics(url: str) -> dict:
        parts = urlsplit(url)
        return {
            "url_length": len(url),
            "dot_count": url.count("."),
            "slash_count": url.count("/"),
            "contains_ip": int(bool(re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url))),
            "uses_https": int(parts.scheme == "https"),
            "has_at_symbol": int("@" in url),
            "path_depth": len([p for p in parts.path.split("/") if p]),
            "query_length": len(parts.query),
            "special_char_count": sum(not c.isalnum() for c in url),
        }

    # ---------------------------
    # LOAD & CLEAN DATA
    # ---------------------------
    df = pd.read_csv("processed_urls.csv")
    df.columns = df.columns.str.lower()

    # FIX: safely map labels
    df["label"] = df["label"].replace({
        "phishing": 1,
        "legitimate": 0,
        -1: 1,
        1.0: 1,
        0.0: 0
    })

    # Drop rows with unknown labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ---------------------------
    # FEATURE MATRIX
    # ---------------------------
    X = pd.DataFrame(
        df["url"].astype(str).apply(extract_url_metrics).tolist()
    )
    y = df["label"]

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_url = st.text_input("🔗 Enter a URL to analyze")

    if st.button("Analyze URL"):
        if not user_url.strip():
            st.warning("Please enter a valid URL.")
        else:
            features = pd.DataFrame(
                [extract_url_metrics(user_url)]
            )

            phishing_prob = model.predict_proba(features)[0][1]

            if phishing_prob >= 0.75:
                st.error(f"🚩 High Risk Phishing URL ({phishing_prob:.2f})")
            elif phishing_prob >= 0.45:
                st.warning(f"⚠️ Suspicious URL ({phishing_prob:.2f})")
            else:
                st.success(f"✅ Likely Safe URL ({phishing_prob:.2f})")

            st.subheader("🔍 Extracted Features")
            st.json(features.to_dict(orient="records")[0])

# ==================================================
# YARA RULE MATCHER
# ==================================================
if module == "YARA Rule Matcher":

    st.header("🧬 YARA Rule Matcher")
    st.write("Matches YARA rules using character-level TF-IDF similarity.")

    # ---------------------------
    # LOAD YARA DATA
    # ---------------------------
    @st.cache_data
    def load_yara_rules():
        df = pd.read_csv("yara_rules_clean.csv")
        return df.dropna(subset=["rule_text"])

    yara_df = load_yara_rules()

    # ---------------------------
    # STRING EXTRACTION
    # ---------------------------
    def extract_strings(rule_text: str) -> str:
        matches = re.findall(r'\$[\w]+\s*=\s*(.+)', rule_text)
        return " ".join(matches).lower()

    yara_df["strings"] = yara_df["rule_text"].apply(extract_strings)
    yara_df = yara_df[yara_df["strings"].str.len() > 15]

    # ---------------------------
    # TF-IDF VECTORIZATION
    # ---------------------------
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 7),
        max_features=30000
    )

    tfidf_matrix = vectorizer.fit_transform(yara_df["strings"])

    # ---------------------------
    # USER INPUT
    # ---------------------------
    indicators = st.text_area(
        "Paste malware strings or commands",
        height=160
    )

    top_k = st.slider("Number of matching rules", 1, 10, 5)

    if st.button("Match Rules"):
        if not indicators.strip():
            st.warning("Please paste some indicators.")
        else:
            query_vec = vectorizer.transform(
                [indicators.lower()]
            )

            scores = cosine_similarity(
                query_vec, tfidf_matrix
            ).flatten()

            yara_df["score"] = scores
            results = yara_df.sort_values(
                "score", ascending=False
            ).head(top_k)

            st.subheader("📌 Recommended YARA Rules")

            for _, row in results.iterrows():
                with st.expander(
                    f"{row.get('rule_name', 'Unnamed Rule')} "
                    f"(Score: {row['score']:.2f})"
                ):
                    st.code(row["rule_text"], language="yara")
