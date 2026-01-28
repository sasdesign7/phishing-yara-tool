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
# PAGE CONFIG (UI)
# --------------------------------------------------
st.set_page_config(
    page_title="Security Analysis Toolkit",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR (UI)
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Security Toolkit")
    st.caption("Threat analysis & detection")
    st.divider()

# --------------------------------------------------
# MAIN HEADER (UI)
# --------------------------------------------------
st.markdown(
    """
    # 🛡️ Security Analysis Toolkit
    **Machine-assisted security analysis tools**
    """
)

tab1, tab2 = st.tabs([
    "🚨 Phishing URL Detector",
    "🧩 YARA Rule Recommendation"
])

# ==================================================
# 🚨 PHISHING URL DETECTOR
# ==================================================
with tab1:
    st.subheader("🚨 Phishing URL Detector")
    st.caption("ML-based detection using structural URL features")

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
    data.rename(columns={"url": "url", "label": "label"}, inplace=True)
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

    clf = GradientBoostingClassifier()
    clf.fit(Xtr, ytr)

    # ---------------------------
    # USER INPUT (UI)
    # ---------------------------
    col1, col2 = st.columns([4, 1])
    with col1:
        user_url = st.text_input("🔗 URL to analyze", placeholder="https://example.com/login")
    with col2:
        analyze = st.button("Analyze", use_container_width=True)

    if analyze:
        if not user_url.strip():
            st.warning("Please enter a URL.")
        else:
            feats = pd.DataFrame([url_features(user_url)])
            prob = clf.predict_proba(feats)[0][1]

            st.divider()
            st.subheader("📊 Risk Assessment")

            st.progress(float(prob))

            if prob > 0.7:
                st.error(f"⚠️ **High Risk Phishing URL** — {prob:.2f}")
            elif prob > 0.4:
                st.warning(f"⚠️ **Medium Risk URL** — {prob:.2f}")
            else:
                st.success(f"✅ **Low Risk URL** — {prob:.2f}")

            with st.expander("🔍 Feature Breakdown"):
                st.json(feats.to_dict(orient="records")[0])

# ==================================================
# 🛡️ YARA RULE RECOMMENDER
# ==================================================
with tab2:
    st.subheader("🧩 YARA Rule Recommendation Tool")
    st.caption("TF-IDF similarity matching against known YARA rules")

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
    def pull_strings(rule):
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
    # USER INPUT (UI)
    # ---------------------------
    indicators = st.text_area(
        "🧬 Malware strings / commands",
        placeholder="cmd.exe /c powershell -enc ...",
        height=160
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of rules", 1, 10, 5)
    with col2:
        recommend = st.button("Recommend", use_container_width=True)

    if recommend:
        if not indicators.strip():
            st.warning("Please paste some malware indicators.")
        else:
            q_vec = vectorizer.transform([indicators.lower()])
            sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

            yara_df["score"] = sims
            results = yara_df.sort_values(
                "score", ascending=False
            ).head(top_k)

            st.divider()
            st.subheader("📌 Recommended YARA Rules")

            for _, row in results.iterrows():
                with st.expander(
                    f"🧩 {row.get('rule_name', 'Unnamed Rule')} — Score {row['score']:.2f}"
                ):
                    st.code(row["rule_text"], language="yara")
