"""
app.py
======
Streamlit Web Application for AI-Based Phishing Detection.
Features:
- State-of-the-Art BERT Transformer Phishing Email Analysis
- Random Forest Machine Learning Baseline
- Interactive Side-by-Side Model Comparison
- URL Phishing Analyzer
- Comparative Benchmark Metrics & Architecture Explanations
"""

import os
import sys
import re
from urllib.parse import urlparse
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Page Configuration
st.set_page_config(
    page_title="AI Phishing Shield (BERT Powered)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-phish {
        background-color: #FEE2E2;
        border: 1px solid #EF4444;
        color: #991B1B;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
    }
    .alert-safe {
        background-color: #DCFCE7;
        border: 1px solid #22C55E;
        color: #166534;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Cache Model Loaders
@st.cache_resource
def load_bert_model():
    from bert_classifier import get_bert_detector
    return get_bert_detector()

@st.cache_resource
def load_baseline_models():
    rf_email = joblib.load('models/rf_email.pkl') if os.path.exists('models/rf_email.pkl') else None
    rf_url = joblib.load('models/rf_url.pkl') if os.path.exists('models/rf_url.pkl') else None
    tfidf = joblib.load('models/tfidf_vectorizer.pkl') if os.path.exists('models/tfidf_vectorizer.pkl') else None
    return rf_email, rf_url, tfidf


def clean_text_for_tfidf(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_email_features_rf(text, tfidf_vec):
    clean = clean_text_for_tfidf(text)
    tfidf_feat = tfidf_vec.transform([clean])
    raw_text = str(text)

    extra = np.array([[
        len(raw_text.split()),
        len(raw_text),
        1 if 'http' in raw_text.lower() else 0,
        len(re.findall(r'http\S+', raw_text)),
        raw_text.count('!'),
        sum(1 for c in raw_text if c.isupper()) / (len(raw_text) + 1),
        1 if any(w in raw_text.lower() for w in ['urgent','verify','account','suspend','click','password','login','security','alert','bank']) else 0
    ]])
    return hstack([tfidf_feat, csr_matrix(extra)])


def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url)
    return pd.DataFrame([{
        'url_length'          : len(url),
        'domain_length'       : len(parsed.netloc),
        'path_length'         : len(parsed.path),
        'num_dots'            : url.count('.'),
        'num_hyphens'         : url.count('-'),
        'num_underscores'     : url.count('_'),
        'num_slashes'         : url.count('/'),
        'num_at'              : url.count('@'),
        'num_question'        : url.count('?'),
        'num_equals'          : url.count('='),
        'num_digits'          : sum(c.isdigit() for c in url),
        'num_params'          : len(parsed.query.split('&')) if parsed.query else 0,
        'has_https'           : 1 if parsed.scheme == 'https' else 0,
        'has_ip'              : 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0,
        'has_suspicious_words': 1 if any(w in url.lower() for w in ['login','verify','bank','secure','account','update','confirm','paypal']) else 0,
        'subdomain_count'     : len(parsed.netloc.split('.')) - 2 if parsed.netloc else 0,
        'shortening_service'  : 1 if any(s in url for s in ['bit.ly','tinyurl','t.co','goo.gl']) else 0,
    }])


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.title("Phishing Shield")
    st.caption("v2.0 — Transformer & ML Engine")

    st.markdown("---")
    st.markdown("### 🧠 Active Architectures")
    st.markdown("""
    - **BERT Transformer (Deep Learning)**
      - Bidirectional Self-Attention
      - 98.6% Accuracy, 100% Recall
    - **Random Forest (Classic ML)**
      - 100 Trees + TF-IDF (5007 features)
      - 96.4% Accuracy
    """)

    st.markdown("---")
    st.info("💡 **Tip**: Traditional models can be fooled by synonym replacement and urgency without obvious keywords. BERT inspects the entire syntactic context.")


# Header
st.markdown('<div class="main-title">🛡️ AI-Powered Phishing Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent cyber threat detection using <b>BERT Transformers</b> and Machine Learning.</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_email, tab_compare, tab_url, tab_metrics = st.tabs([
    "📧 Email Threat Detector",
    "⚖️ Side-by-Side Model Comparison",
    "🔗 URL Threat Detector",
    "📊 Benchmarks & Architecture"
])

# ----------------- TAB 1: EMAIL DETECTION -----------------
with tab_email:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        st.subheader("Analyze Email Content")

        # Sample quick loaders
        st.markdown("**Quick Test Samples:**")
        s1, s2, s3 = st.columns(3)
        sample_text = ""

        if s1.button("🚨 Bank Phishing"):
            st.session_state['email_input'] = (
                "URGENT SECURITY ALERT: Your online banking account has been temporarily locked "
                "due to multiple suspicious failed login attempts. You must confirm your credentials "
                "within 24 hours at http://192.168.1.100/verify-account or your account will be permanently closed."
            )
        if s2.button("⚠️ Cloud Service Spoof"):
            st.session_state['email_input'] = (
                "Important notice: Your Microsoft 365 cloud backup sync encountered a timeout. "
                "Please review your admin portal credentials to resume your corporate mailbox service."
            )
        if s3.button("✅ Safe Work Email"):
            st.session_state['email_input'] = (
                "Hi Sarah, attached is the revised agenda for tomorrow's sprint retrospective. "
                "Please take a look at the slide deck before 2 PM. Let me know if you'd like to add any discussion points."
            )

        email_text = st.text_area(
            "Paste full email body or subject line:",
            value=st.session_state.get('email_input', ''),
            height=200,
            placeholder="Dear customer, your account requires immediate verification..."
        )

        model_choice = st.radio(
            "Select Detection Engine:",
            ["🤖 BERT Transformer (Deep Learning - Recommended: 98.6% Accuracy)",
             "🌲 Random Forest (Classic TF-IDF Baseline - 96.4% Accuracy)"],
            index=0
        )

        analyze_btn = st.button("🔍 Run Email Analysis", type="primary", use_container_width=True)

    with col_meta:
        st.subheader("Model Information")
        if "BERT" in model_choice:
            st.markdown("""
            <div class="metric-card">
                <h4>🤖 BERT Transformer</h4>
                <p><b>Type:</b> Pretrained / Fine-tuned Transformer</p>
                <p><b>Strengths:</b> Deep bidirectional context, detects subtle social engineering, handles evasive paraphrasing.</p>
                <p><b>Test Accuracy:</b> <b style="color:#22C55E;">98.60%</b></p>
                <p><b>Phishing Recall:</b> <b style="color:#22C55E;">100.00%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>🌲 Random Forest</h4>
                <p><b>Type:</b> 100 Decision Trees Ensemble</p>
                <p><b>Features:</b> 5,000 TF-IDF N-grams + 7 Meta features</p>
                <p><b>Test Accuracy:</b> 96.43%</p>
                <p><b>Phishing Recall:</b> 97.06%</p>
            </div>
            """, unsafe_allow_html=True)

    # Analysis Results Section
    if analyze_btn:
        if not email_text.strip():
            st.warning("⚠️ Please paste an email text to analyze.")
        else:
            with st.spinner("Analyzing semantics with selected model..."):
                if "BERT" in model_choice:
                    bert = load_bert_model()
                    res = bert.predict(email_text)

                    st.markdown("---")
                    st.subheader("Analysis Verdict")

                    if res['is_phishing']:
                        st.markdown(f"""
                        <div class="alert-phish">
                            🚨 <b>THREAT DETECTED: {res['risk_level']}</b><br>
                            This email exhibits high-confidence characteristics of a malicious phishing attack.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-safe">
                            ✅ <b>VERDICT: {res['risk_level']}</b><br>
                            This email appears legitimate based on natural language analysis.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Phishing Probability", f"{res['phishing_prob']:.2%}")
                    m2.metric("Legitimate Probability", f"{res['legit_prob']:.2%}")
                    m3.metric("Model Confidence", f"{res['confidence']:.2%}")

                    st.progress(res['phishing_prob'], text=f"Phishing Risk Score: {res['phishing_prob']*100:.1f} / 100")

                    if res['cues']:
                        st.markdown("#### 🚩 Key Detected Phishing Signals:")
                        for cue in res['cues']:
                            st.write(f"- ⚠️ {cue}")
                else:
                    rf_email, _, tfidf = load_baseline_models()
                    if rf_email is None or tfidf is None:
                        st.error("Baseline model files not found in models/.")
                    else:
                        feat = extract_email_features_rf(email_text, tfidf)
                        pred = rf_email.predict(feat)[0]
                        probs = rf_email.predict_proba(feat)[0]
                        phish_prob = float(probs[1])

                        st.markdown("---")
                        st.subheader("Analysis Verdict (Random Forest)")
                        if pred == 1:
                            st.markdown(f"""
                            <div class="alert-phish">
                                🚨 <b>THREAT DETECTED: PHISHING EMAIL</b><br>
                                Random Forest classified this text as Phishing with {phish_prob:.1%} confidence.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="alert-safe">
                                ✅ <b>VERDICT: LEGITIMATE EMAIL</b><br>
                                Random Forest classified this text as Legitimate with {(1-phish_prob):.1%} confidence.
                            </div>
                            """, unsafe_allow_html=True)

                        m1, m2 = st.columns(2)
                        m1.metric("Phishing Probability", f"{phish_prob:.2%}")
                        m2.metric("Legitimate Probability", f"{(1-phish_prob):.2%}")
                        st.progress(phish_prob, text=f"Phishing Risk Score: {phish_prob*100:.1f} / 100")


# ----------------- TAB 2: SIDE-BY-SIDE COMPARISON -----------------
with tab_compare:
    st.subheader("🔬 Live Dual Model Comparison (BERT vs Random Forest)")
    st.write("Input an email to see both architectures evaluate the text simultaneously and observe how BERT handles subtle phrasing.")

    comp_text = st.text_area(
        "Email text for side-by-side benchmark:",
        value="Dear team member, your password will expire in 6 hours due to periodic IT policy. Update it immediately to prevent suspension.",
        height=140
    )

    if st.button("⚡ Compare Models Side-by-Side", type="primary"):
        c1, c2 = st.columns(2)

        bert = load_bert_model()
        bert_res = bert.predict(comp_text)

        rf_email, _, tfidf = load_baseline_models()
        rf_feat = extract_email_features_rf(comp_text, tfidf)
        rf_pred = rf_email.predict(rf_feat)[0]
        rf_prob = float(rf_email.predict_proba(rf_feat)[0][1])

        with c1:
            st.markdown("### 🤖 BERT Transformer (Deep Learning)")
            if bert_res['is_phishing']:
                st.error(f"🚨 {bert_res['prediction']} ({bert_res['risk_level']})")
            else:
                st.success(f"✅ {bert_res['prediction']} ({bert_res['risk_level']})")
            st.metric("Phishing Probability", f"{bert_res['phishing_prob']:.2%}")
            st.progress(bert_res['phishing_prob'])
            st.write("**Key Advantage:** Evaluates semantic urgency and intent across full sentences.")

        with c2:
            st.markdown("### 🌲 Random Forest (TF-IDF Baseline)")
            if rf_pred == 1:
                st.error("🚨 PHISHING DETECTED")
            else:
                st.success("✅ LEGITIMATE EMAIL")
            st.metric("Phishing Probability", f"{rf_prob:.2%}")
            st.progress(rf_prob)
            st.write("**Limitation:** Relies strictly on isolated word frequencies and exact token matches.")


# ----------------- TAB 3: URL THREAT DETECTOR -----------------
with tab_url:
    st.subheader("🔗 URL Phishing Analyzer")
    st.write("Detect malicious, deceptive, and spoofed website links using lexical & structural feature extraction.")

    url_samples = st.columns(3)
    if url_samples[0].button("⚠️ IP Phishing Link"):
        st.session_state['url_input'] = "http://192.168.1.1/paypal/login.php?cmd=_login"
    if url_samples[1].button("⚠️ Typosquat Link"):
        st.session_state['url_input'] = "http://www.micros0ft-security-update-portal.com/login"
    if url_samples[2].button("✅ Safe Website"):
        st.session_state['url_input'] = "https://www.google.com/search?q=cybersecurity"

    url_input = st.text_input("Enter URL to analyze:", value=st.session_state.get('url_input', ''), placeholder="https://example.com/login")

    if st.button("🔍 Analyze URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            _, rf_url, _ = load_baseline_models()
            if rf_url is None:
                st.error("URL Random Forest model not found.")
            else:
                url_features = extract_url_features(url_input)
                pred = rf_url.predict(url_features)[0]
                probs = rf_url.predict_proba(url_features)[0]
                phish_prob = float(probs[1])

                st.markdown("---")
                if pred == 1:
                    st.markdown(f"""
                    <div class="alert-phish">
                        🚨 <b>MALICIOUS URL DETECTED!</b><br>
                        Confidence: <b>{phish_prob:.1%}</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-safe">
                        ✅ <b>URL APPEARS SAFE!</b><br>
                        Confidence: <b>{(1-phish_prob):.1%}</b>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.write("#### Extracted URL Structural Attributes:")
                st.dataframe(url_features.T.rename(columns={0: "Feature Value"}))


# ----------------- TAB 4: BENCHMARKS & ARCHITECTURE -----------------
with tab_metrics:
    st.subheader("📊 Performance Benchmarks & Architecture Breakdown")

    # Load results CSV if available
    results_path = 'models/model_comparison_results.csv'
    if os.path.exists(results_path):
        bench_df = pd.read_csv(results_path)
        st.write("### Model Evaluation Summary (Test Dataset)")
        st.dataframe(bench_df.style.highlight_max(subset=['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC'], color='#D1FAE5'))

    chart_path = 'models/accuracy_comparison_chart.png'
    if os.path.exists(chart_path):
        st.image(chart_path, caption="Comparative Metrics: BERT vs Traditional Baselines", use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Why BERT Achieves Superior Accuracy Over TF-IDF")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        #### ❌ Limitations of TF-IDF + Classic ML:
        1. **Bag-of-Words Fallacy:** Treats words as unordered bags, missing context.
        2. **Easily Evaded:** Attackers bypass TF-IDF by replacing trigger words with synonyms (e.g., swapping *"verify password"* for *"confirm credentials"*).
        3. **No Tone or Intent Understanding:** Fails to differentiate between actual business urgency vs predatory social engineering pressure.
        """)

    with col_b:
        st.markdown("""
        ####  How BERT Solves It:
        1. **Bidirectional Self-Attention:** Evaluates relationships between every word and every other word in the text simultaneously.
        2. **Subword WordPiece Tokenization:** Resilient against obfuscation, typosquatting, and leetspeak.
        3. **Semantic Intent Comprehension:** Understands coercive phishing context even when entirely novel vocabulary is used.
        """)
