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
# APP SETTINGS
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
    st.caption("Detects suspicious URLs using machine learning.")

    # ---------------------------
    # URL FEATURE EXTRACTION
    # ---------------------------
    def extract_url_metrics(link: str) -> dict:
        parts = urlsplit(link)

        return {
            "length_total": len(link),
            "num_dots": link.count("."),
            "num_slashes": link.count("/"),
            "contains_ip": int(bool(re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", link))),
            "uses_https": int(parts.scheme == "https"),
            "has_at_symbol": int("@" in link),
            "path_segments": len([p for p in parts.path.split("/") if p]),
            "query_size": len(parts.query),
            "non_alnum_chars": sum(not c.isalnum() for c in link),
        }

    # ---------------------------
    # LOAD & PREP DATA
    # ---------------------------
    df = pd.read_csv("processed_urls.csv")
    df.columns = df.columns.str.lower()

    df["label"] = df["label"].replace(
        {"phishing": 1, "legitimate": 0, -1: 1}
    ).astype(int)

    feature_df = pd.DataFrame(
        df["url"].astype(str).apply(extract_url_metrics).tolist()
    )
    labels = df["label"]

    # ---------------------------
    # MODEL TRAINING
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, labels, test_size=0.25, random_state=21
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=21
    )
    model.fit(X_train, y_train)

    # ---------------------------
    # USER INPUT
    # ---------------------------
    input_url = st.text_input("🔗 Enter a URL")

    if st.button("Scan URL"):
        if not input_url.strip():
            st.warning("URL cannot be empty.")
        else:
            input_features = pd.DataFrame(
                [extract_url_metrics(input_url)]
            )
            risk_score = model.predict_proba(input_features)[0][1]

            if risk_score >= 0.75:
                st.error(f"🚩 Likely Phishing ({risk_score:.2f})")
            elif risk_score >= 0.45:
                st.warning(f"⚠️ Suspicious ({risk_score:.2f})")
            else:
                st.success(f"✅ Likely Safe ({risk_score:.2f})")

            st.subheader("🔍 Extracted Features")
            st.table(input_features)

# ==================================================
# YARA RULE MATCHER
# ==================================================
if module == "YARA Rule Matcher":

    st.header("🧬 YARA Rule Matcher")
    st.caption("Finds relevant YARA rules based on string similarity.")

    # ---------------------------
    # LOAD RULES
    # ---------------------------
    @st.cache_data
    def read_yara_rules():
        rules_df = pd.read_csv("yara_rules_clean.csv")
        return rules_df.dropna(subset=["rule_text"])

    yara_rules = read_yara_rules()

    # ---------------------------
    # STRING CLEANING
    # ---------------------------
    def extract_rule_strings(text):
        found = re.findall(r'\$[\w]+\s*=\s*(.+)', text)
        return " ".join(found).lower()

    yara_rules["extracted_strings"] = yara_rules["rule_text"].apply(
        extract_rule_strings
    )
    yara_rules = yara_rules[
        yara_rules["extracted_strings"].str.len() > 15
    ]

    # ---------------------------
    # VECTORIZE
    # ---------------------------
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 7),
        max_features=30000
    )
    rule_vectors = tfidf.fit_transform(
        yara_rules["extracted_strings"]
    )

    # ---------------------------
    # USER INPUT
    # ---------------------------
    malware_input = st.text_area(
        "Paste suspicious strings or commands",
        height=160
    )

    num_results = st.slider(
        "Top matches",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Match Rules"):
        if not malware_input.strip():
            st.warning("Please provide input strings.")
        else:
            query_vec = tfidf.transform([malware_input.lower()])
            similarity_scores = cosine_similarity(
                query_vec, rule_vectors
            ).flatten()

            yara_rules["similarity"] = similarity_scores
            top_matches = yara_rules.sort_values(
                "similarity", ascending=False
            ).head(num_results)

            st.subheader("📌 Matching YARA Rules")

            for _, rule in top_matches.iterrows():
                with st.expander(
                    f"{rule.get('rule_name', 'Unnamed')} "
                    f"(Score: {rule['similarity']:.2f})"
                ):
                    st.code(rule["rule_text"], language="yara")
