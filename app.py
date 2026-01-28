import streamlit as st
import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Security Analysis Toolkit",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Security Analysis Toolkit")

tool = st.sidebar.radio(
    "Select Tool",
    ["🚨 Phishing URL Detector", "🛡️ YARA Rule Recommendation Tool"]
)

# ==================================================
# 🚨 PHISHING URL DETECTOR
# ==================================================
if tool == "🚨 Phishing URL Detector":

    st.header("🚨 Phishing URL Detector")
    st.write("Machine-learning based detection using structural URL features.")

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    def url_features(u: str) -> dict:
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

    # normalize column names
    data.columns = data.columns.str.lower().str.strip()

    if "url" not in data.columns:
        st.error(f"❌ URL column not found. Found: {list(data.columns)}")
        st.stop()

    if "label" not in data.columns:
        st.error(f"❌ Label column not found. Found: {list(data.columns)}")
        st.stop()

    # normalize labels
    data["label"] = (
        data["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    data["label"] = data["label"].map({
        "phishing": 1,
        "malicious": 1,
        "-1": 1,
        "1": 1,
        "legitimate": 0,
        "benign": 0,
        "0": 0
    })

    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    # ---------------------------
    # FEATURE MATRIX
    # ---------------------------
    X = pd.DataFrame(
        [url_features(u) for u in data["url"].astype(str)]
    )
    y = data["label"]

    # ---------------------------
    # TRAIN MODEL (CACHED)
    # ---------------------------
    @st.cache_resource
    def train_model(X, y):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=7
        )
        clf = GradientBoostingClassifier()
        clf.fit(Xtr, ytr)
        return clf

    clf = train_model(X, y)

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_url = st.text_input("🔗 Enter a URL to analyze")

    if st.button("Analyze URL"):
        if not user_url.strip():
            st.warning("Please enter a URL.")
        else:
            feats = pd.DataFrame([url_features(user_url)])
            prob = clf.predict_proba(feats)[0][1]

            if prob > 0.7:
                st.error(f"⚠️ High Risk Phishing URL ({prob:.2f})")
            elif prob > 0.4:
                st.warning(f"⚠️ Medium Risk URL ({prob:.2f})")
            else:
                st.success(f"✅ Low Risk URL ({prob:.2f})")

            st.subheader("🔍 Feature Breakdown")
            st.json(feats.to_dict(orient="records")[0])

# ==================================================
# 🛡️ YARA RULE RECOMMENDER
# ==================================================
if tool == "🛡️ YARA Rule Recommendation Tool":

    st.header("🛡️ YARA Rule Recommendation Tool")
    st.write("Recommends YARA rules using TF-IDF similarity on rule strings.")

    # ---------------------------
    # LOAD YARA DATA
    # ---------------------------
    @st.cache_data
    def load_yara():
        df = pd.read_csv("yara_rules_clean.csv")
        df = df.dropna(subset=["rule_text"])
        return df

    yara_df = load_yara()

    # ---------------------------
    # STRING EXTRACTION
    # ---------------------------
    def pull_strings(rule: str) -> str:
        matches = re.findall(r'\$[a-zA-Z0-9_]+\s*=\s*(.+)', rule)
        return " ".join(matches).lower()

    yara_df["strings"] = yara_df["rule_text"].apply(pull_strings)
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
        "Paste malware strings / commands",
        height=180
    )

    top_k = st.slider("Number of rules", 1, 10, 5)

    if st.button("Recommend Rules"):
        if not indicators.strip():
            st.warning("Please paste some malware indicators.")
        else:
            q_vec = vectorizer.transform([indicators.lower()])
            sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

            yara_df["score"] = sims
            results = yara_df.sort_values(
                "score", ascending=False
            ).head(top_k)

            st.subheader("📌 Recommended YARA Rules")

            for _, row in results.iterrows():
                with st.expander(
                    f"🧩 {row.get('rule_name', 'Unnamed Rule')} "
                    f"(Score: {row['score']:.2f})"
                ):
                    st.code(row["rule_text"], language="yara")
