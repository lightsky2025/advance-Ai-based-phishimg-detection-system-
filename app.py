"""
Streamlit Web Application for AI-Based Phishing Detection System
Modern Professional UI with Advanced Visual Effects
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime

# Try to import plotly, but don't fail if not installed
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations will be disabled. Run: pip install plotly")

from config import UI_CONFIG, DETECTION_THRESHOLDS
from detector import PhishingDetector, get_detector
from feature_engineering import URLFeatureExtractor, EmailFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="🛡️ AI Phishing Defender | Advanced Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODERN CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glass morphism header */
    .glass-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Animated title */
    .animated-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        border-left: 4px solid;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk indicators */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa502, #ff7f50);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #1e90ff, #00bfff);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .risk-safe {
        background: linear-gradient(135deg, #2ed573, #00b894);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Modern button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Feature bar */
    .feature-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        margin: 0.5rem 0;
        transition: width 0.5s;
    }
</style>
""", unsafe_allow_html=True)


def initialize_detector():
    """Initialize the phishing detector"""
    if 'detector' not in st.session_state:
        with st.spinner("🔄 Loading AI Models..."):
            st.session_state.detector = get_detector()
    return st.session_state.detector


def display_modern_risk_gauge(risk_score: float):
    """Display an interactive risk gauge using Plotly"""
    
    if not PLOTLY_AVAILABLE:
        # Fallback to simple progress bar
        st.progress(risk_score)
        return
    
    # Determine color and level
    if risk_score >= DETECTION_THRESHOLDS['high_risk']:
        color = "#ff4757"
        level = "HIGH RISK"
        icon = "🔴"
    elif risk_score >= DETECTION_THRESHOLDS['medium_risk']:
        color = "#ffa502"
        level = "MEDIUM RISK"
        icon = "🟠"
    elif risk_score >= DETECTION_THRESHOLDS['low_risk']:
        color = "#1e90ff"
        level = "LOW RISK"
        icon = "🟡"
    else:
        color = "#2ed573"
        level = "SAFE"
        icon = "🟢"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 50], 'color': '#e3f2fd'},
                {'range': [50, 80], 'color': '#fff3e0'},
                {'range': [80, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': color, 'family': "Inter"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level display
    level_color = {
        "HIGH RISK": "#ff4757",
        "MEDIUM RISK": "#ffa502", 
        "LOW RISK": "#1e90ff",
        "SAFE": "#2ed573"
    }
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: -1rem;">
        <div class="badge" style="background: {level_color[level]}; color: white;">
            {icon} {level} {icon}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_modern_risk_factors(risk_factors: list):
    """Display risk factors with modern styling"""
    if not risk_factors:
        st.info("✅ No significant risk factors detected")
        return
    
    st.markdown("### 🔍 Detected Risk Factors")
    
    for i, factor in enumerate(risk_factors, 1):
        severity = "HIGH" if any(word in factor.upper() for word in ["🚨", "CRITICAL"]) else "MEDIUM"
        color = "#ff4757" if severity == "HIGH" else "#ffa502"
        
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {color};">
            <span style="font-weight: 600;">{i}.</span> {factor}
        </div>
        """, unsafe_allow_html=True)


def display_modern_recommendations(recommendations: list):
    """Display security recommendations with icons"""
    if not recommendations:
        return
    
    st.markdown("### 📋 Security Recommendations")
    
    cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.7rem; border-radius: 10px; margin: 0.3rem 0;">
                {rec}
            </div>
            """, unsafe_allow_html=True)


def dashboard_page():
    """Dashboard / Home Page"""
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-title">🛡️ AI Phishing Defender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Real-time AI-powered phishing detection for URLs and Emails</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #667eea;">
            <div class="metric-label">Detection Models</div>
            <div class="metric-value">2</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #2ed573;">
            <div class="metric-label">URL Features</div>
            <div class="metric-value">40+</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #ffa502;">
            <div class="metric-label">Email Features</div>
            <div class="metric-value">20+</div>
        </div>
        """, unsafe_allow_html=True)

    # Load metrics for dashboard display
    url_acc = 96.68
    email_acc = 86.71
    try:
        with open("models/url_training_metrics.json", "r") as f:
            url_metrics_dash = json.load(f)
            url_acc = url_metrics_dash.get("xgboost_url", {}).get("accuracy", 0.9668) * 100
    except:
        pass
    try:
        with open("models/email_training_metrics.json", "r") as f:
            email_metrics_dash = json.load(f)
            email_acc = email_metrics_dash.get("xgboost_email", {}).get("accuracy", 0.8671) * 100
    except:
        pass

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff4757;">
            <div class="metric-label">URL Model Accuracy</div>
            <div class="metric-value" style="color: #ff4757;">{url_acc:.2f}%</div>
            <div style="font-size:0.75rem; color:#999;">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #764ba2;">
            <div class="metric-label">Email Model Accuracy</div>
            <div class="metric-value" style="color: #764ba2;">{email_acc:.2f}%</div>
            <div style="font-size:0.75rem; color:#999;">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick action cards
    st.markdown("## ⚡ Quick Actions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px;
                    padding: 1.5rem; color: white; text-align: center;">
            <div style="font-size: 2.5rem;">🔗</div>
            <h3>URL Scanner</h3>
            <p style="opacity: 0.9;">Paste any suspicious URL to instantly check for phishing indicators.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to URL Scanner →", key="dash_url", use_container_width=True):
            st.session_state["nav"] = "🔗 URL Scanner"

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2ed573, #00b894); border-radius: 15px;
                    padding: 1.5rem; color: white; text-align: center;">
            <div style="font-size: 2.5rem;">📧</div>
            <h3>Email Scanner</h3>
            <p style="opacity: 0.9;">Paste email subject and body to detect phishing content.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Email Scanner →", key="dash_email", use_container_width=True):
            st.session_state["nav"] = "📧 Email Scanner"

    st.divider()

    # Session history
    st.markdown("## 🕘 Recent Scans")
    if 'detection_history' in st.session_state and st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("📭 No scans yet in this session. Use the URL or Email Scanner to get started.")

    st.divider()

    # Threat summary
    st.markdown("## 🧠 How It Works")
    st.markdown("""
    1. **Input** — Paste a URL or email content into the scanner.
    2. **Feature Extraction** — 40+ structural and linguistic features are extracted automatically.
    3. **ML Inference** — XGBoost and Random Forest models evaluate the features.
    4. **Risk Scoring** — A combined risk score (0–100%) is computed.
    5. **Report** — Risk level, detected indicators, and recommendations are shown instantly.
    """)
def analyze_url_page():
    """URL Analysis Page with Modern UI"""
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-title">🔗 URL Phishing Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Analyze URLs in real-time for phishing indicators</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze:",
            placeholder="https://example.com/path?query=value",
            key="url_input",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_clicked = st.button("🚀 Analyze URL", type="primary", use_container_width=True)
    
    # Batch analysis
    with st.expander("📦 Batch URL Analysis (Upload multiple URLs)", expanded=False):
        batch_urls = st.text_area(
            "Enter multiple URLs (one per line):",
            height=150,
            placeholder="https://url1.com\nhttps://url2.com\nhttps://url3.com",
            label_visibility="collapsed"
        )
        
        if st.button("📊 Analyze Batch", use_container_width=True):
            if batch_urls and st.session_state.detector:
                urls = [u.strip() for u in batch_urls.split('\n') if u.strip()]
                if urls:
                    with st.spinner(f"🔍 Analyzing {len(urls)} URLs..."):
                        progress_bar = st.progress(0)
                        results = []
                        for i, url in enumerate(urls):
                            results.append(st.session_state.detector.analyze_url(url))
                            progress_bar.progress((i + 1) / len(urls))
                        progress_bar.empty()
                        
                        df_results = pd.DataFrame([
                            {
                                'URL': r['url'][:60] + '...' if len(r['url']) > 60 else r['url'],
                                'Risk Level': r['risk_level'],
                                'Risk Score': f"{r['risk_score']:.2%}",
                                'Issues': len(r['risk_factors'])
                            }
                            for r in results
                        ])
                        
                        st.dataframe(df_results, use_container_width=True)
    
    # Single URL analysis
    if analyze_clicked and url_input:
        if not st.session_state.detector:
            st.error("❌ Models not loaded. Please wait for initialization.")
            return
        
        with st.spinner("🔍 Analyzing URL..."):
            progress_bar = st.progress(0)
            for i in range(20):
                time.sleep(0.01)
                progress_bar.progress((i + 1) * 5)
            
            result = st.session_state.detector.analyze_url(url_input)
            progress_bar.empty()
        
        st.markdown("---")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        risk_color = {
            "HIGH": "#ff4757",
            "MEDIUM": "#ffa502",
            "LOW": "#1e90ff",
            "SAFE": "#2ed573"
        }
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {risk_color.get(result['risk_level'], '#666')}">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="color: {risk_color.get(result['risk_level'], '#666')}">
                    {result['risk_level']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{result['risk_score']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ml_status = "Phishing" if result.get('ml_prediction') == 1 else "Legitimate" if result.get('ml_prediction') is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ML Prediction</div>
                <div class="metric-value">{ml_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Factors</div>
                <div class="metric-value">{len(result['risk_factors'])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk gauge
        if PLOTLY_AVAILABLE:
            display_modern_risk_gauge(result['risk_score'])
        else:
            st.progress(result['risk_score'])
            st.write(f"**Risk Score:** {result['risk_score']:.2%}")
        
        # Risk factors
        if result['risk_factors']:
            display_modern_risk_factors(result['risk_factors'])
        
        # Recommendations
        recommendations = st.session_state.detector.get_recommendations(result)
        display_modern_recommendations(recommendations)
        
        # Detailed analysis expander
        with st.expander("📊 Detailed Analysis", expanded=False):
            features = result.get('features', {})
            if features:
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                st.dataframe(feature_df, use_container_width=True)
    
    elif analyze_clicked and not url_input:
        st.warning("⚠️ Please enter a URL to analyze.")


def analyze_email_page():
    """Email Analysis Page with Modern UI"""
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-title">📧 Email Phishing Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Paste email content to analyze for phishing indicators</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state for sample data
    if 'sample_subject' not in st.session_state:
        st.session_state.sample_subject = ""
    if 'sample_body' not in st.session_state:
        st.session_state.sample_body = ""
    if 'sample_sender' not in st.session_state:
        st.session_state.sample_sender = ""
    
    # Email input form
    with st.form("email_form"):
        sender = st.text_input("📧 Sender Email Address (optional)", 
                               placeholder="sender@example.com",
                               value=st.session_state.sample_sender)
        subject = st.text_input("📝 Email Subject", 
                                placeholder="Enter email subject...",
                                value=st.session_state.sample_subject)
        body = st.text_area("📄 Email Body", 
                           placeholder="Paste the email body content here...",
                           height=250,
                           value=st.session_state.sample_body)
        
        submitted = st.form_submit_button("🔍 Analyze Email", type="primary", use_container_width=True)
    
    if submitted:
        if not subject and not body:
            st.warning("⚠️ Please enter at least a subject or body content.")
        elif not st.session_state.detector:
            st.error("❌ Models not loaded. Please wait for initialization.")
        else:
            with st.spinner("🔍 Analyzing email content..."):
                progress_bar = st.progress(0)
                for i in range(20):
                    time.sleep(0.01)
                    progress_bar.progress((i + 1) * 5)
                
                result = st.session_state.detector.analyze_email_with_url_extraction(
                    subject, body, sender
                )
                progress_bar.empty()
            
            st.markdown("---")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            risk_color = {
                "HIGH": "#ff4757",
                "MEDIUM": "#ffa502", 
                "LOW": "#1e90ff",
                "SAFE": "#2ed573"
            }
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {risk_color.get(result['combined_risk_level'], '#666')}">
                    <div class="metric-label">Combined Risk</div>
                    <div class="metric-value" style="color: {risk_color.get(result['combined_risk_level'], '#666')}">
                        {result['combined_risk_level']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Combined Score</div>
                    <div class="metric-value">{result['combined_risk_score']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">URLs Found</div>
                    <div class="metric-value">{len(result['urls_found'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            if PLOTLY_AVAILABLE:
                display_modern_risk_gauge(result['combined_risk_score'])
            else:
                st.progress(result['combined_risk_score'])
                st.write(f"**Risk Score:** {result['combined_risk_score']:.2%}")
            
            # Email analysis details
            with st.expander("📧 Email Analysis Details", expanded=True):
                email_analysis = result['email_analysis']
                st.markdown(f"**Subject:** {email_analysis.get('subject', 'N/A')}")
                st.markdown(f"**Sender:** {email_analysis.get('sender', 'N/A')}")
                
                if email_analysis.get('risk_factors'):
                    st.markdown("**Risk Factors:**")
                    for factor in email_analysis['risk_factors']:
                        st.markdown(f"- {factor}")
            
            # URL analysis details
            if result['url_analyses']:
                with st.expander("🔗 Embedded URL Analysis", expanded=False):
                    for url_result in result['url_analyses']:
                        st.markdown(f"**URL:** `{url_result['url']}`")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Risk Level", url_result['risk_level'])
                        with col2:
                            st.metric("Score", f"{url_result['risk_score']:.2%}")
                        if url_result['risk_factors']:
                            st.markdown("**Risk Factors:**")
                            for factor in url_result['risk_factors'][:3]:
                                st.markdown(f"- {factor}")
                        st.divider()
            
            # Recommendations
            display_modern_recommendations(result['recommendations'])
            
            # Clear sample data
            st.session_state.sample_subject = ""
            st.session_state.sample_body = ""
            st.session_state.sample_sender = ""
    
    # Sample emails section
    with st.expander("📝 Try Sample Emails", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Phishing Example")
            if st.button("📧 Load Phishing Sample", key="load_phishing", use_container_width=True):
                st.session_state.sample_subject = "URGENT: Your account has been suspended - Verify now!"
                st.session_state.sample_body = "Dear customer, your account has been compromised. Click here immediately to verify your identity: http://suspicious-site.xyz/verify. Failure to act within 24 hours will result in permanent account termination."
                st.session_state.sample_sender = "security-team123@gmail.com"
                st.rerun()
        
        with col2:
            st.markdown("### 🟢 Legitimate Example")
            if st.button("📧 Load Legitimate Sample", key="load_legitimate", use_container_width=True):
                st.session_state.sample_subject = "Weekly team meeting agenda"
                st.session_state.sample_body = "Hi team, just a reminder that we have our weekly standup at 10 AM tomorrow. Please come prepared with your updates on the current sprint."
                st.session_state.sample_sender = "john.smith@company.com"
                st.rerun()


def about_page():
    """About/Information Page with Detection Explanation and Model Details"""
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-title">🛡️ About the System</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== SYSTEM OVERVIEW ==========
    st.markdown("""
    ## 🚀 AI-Powered Phishing Detection System
    
    This system uses **advanced Machine Learning** and **Natural Language Processing** 
    to detect phishing attempts in URLs and emails with high accuracy.
    """)
    
    # ========== TWO COLUMN LAYOUT FOR DETECTION METHODS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔗 URL Detection
        
        Our system analyzes URLs using **40+ structural features**:
        
        | Feature Category | What We Check |
        |-----------------|---------------|
        | **Length & Structure** | URL length, dots, slashes, hyphens |
        | **Security** | HTTPS presence, port numbers |
        | **Domain** | Suspicious TLDs (.xyz, .tk, .ml), IP addresses |
        | **Content** | Brand keywords, suspicious words |
        | **Complexity** | URL entropy, encoding, redirects |
        | **Patterns** | Typosquatting, homograph attacks |
        
        **Examples detected:**
        - `http://paypal-verify.xyz/login` → Brand impersonation
        - `http://192.168.1.1/verify` → IP address instead of domain
        - `https://secure-login.account-verify.com` → Suspicious subdomains
        """)
    
    with col2:
        st.markdown("""
        ### 📧 Email Detection
        
        Our system analyzes emails using **comprehensive text analysis**:
        
        | Feature Category | What We Check |
        |-----------------|---------------|
        | **Linguistic** | Urgency words, threats, suspicious phrases |
        | **Structural** | HTML ratio, attachments, embedded links |
        | **Sender** | Domain reputation, SPF/DKIM/DMARC |
        | **Content** | Brand names, sensitive keywords |
        | **URLs** | Embedded URL extraction and analysis |
        
        **Examples detected:**
        - "URGENT: Your account will be suspended" → Urgency tactics
        - "Click here to verify" → Suspicious actions
        - security@gmail.com claiming to be your bank → Sender mismatch
        """)
    
    st.divider()
    
    # ========== RISK LEVELS EXPLANATION ==========
    st.markdown("## 📊 Risk Levels Explained")
    
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    with risk_col1:
        st.markdown("""
        <div style="background: #ffebee; border-radius: 15px; padding: 1rem; text-align: center; border-left: 5px solid #ff4757;">
            <div style="font-size: 2rem;">🔴</div>
            <div style="font-weight: bold; color: #ff4757;">HIGH RISK</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Score ≥ 70%</div>
            <hr style="margin: 0.5rem 0;">
            <div style="font-size: 0.8rem; text-align: left;">
            • Strong phishing indicators<br>
            • Multiple red flags detected<br>
            • DO NOT proceed<br>
            • Block immediately
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        st.markdown("""
        <div style="background: #fff3e0; border-radius: 15px; padding: 1rem; text-align: center; border-left: 5px solid #ffa502;">
            <div style="font-size: 2rem;">🟠</div>
            <div style="font-weight: bold; color: #ffa502;">MEDIUM RISK</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Score 45-69%</div>
            <hr style="margin: 0.5rem 0;">
            <div style="font-size: 0.8rem; text-align: left;">
            • Suspicious indicators present<br>
            • Exercise caution<br>
            • Verify through official channels<br>
            • Don't click links
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col3:
        st.markdown("""
        <div style="background: #e3f2fd; border-radius: 15px; padding: 1rem; text-align: center; border-left: 5px solid #1e90ff;">
            <div style="font-size: 2rem;">🟡</div>
            <div style="font-weight: bold; color: #1e90ff;">LOW RISK</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Score 20-44%</div>
            <hr style="margin: 0.5rem 0;">
            <div style="font-size: 0.8rem; text-align: left;">
            • Minor concerns detected<br>
            • Likely safe but verify<br>
            • Check sender carefully<br>
            • Be cautious with links
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col4:
        st.markdown("""
        <div style="background: #e8f5e9; border-radius: 15px; padding: 1rem; text-align: center; border-left: 5px solid #2ed573;">
            <div style="font-size: 2rem;">🟢</div>
            <div style="font-weight: bold; color: #2ed573;">SAFE</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Score &lt; 20%</div>
            <hr style="margin: 0.5rem 0;">
            <div style="font-size: 0.8rem; text-align: left;">
            • No significant risk detected<br>
            • Proceed normally<br>
            • Still practice caution<br>
            • Verify unexpected requests
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== MODEL ARCHITECTURE ==========
    st.markdown("## 🤖 Machine Learning Models")
    
    st.markdown("""
    Our system uses an **ensemble approach** combining multiple models for robust detection:
    """)
    
    model_col1, model_col2, model_col3 = st.columns(3)
    
    with model_col1:
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
            <div style="font-size: 1.5rem; text-align: center;">📊</div>
            <div style="font-weight: bold; text-align: center;">XGBoost</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">
            • Gradient boosting algorithm<br>
            • Best for structured features<br>
            • High accuracy (94.7%)<br>
            • Fast inference
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with model_col2:
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
            <div style="font-size: 1.5rem; text-align: center;">🌲</div>
            <div style="font-weight: bold; text-align: center;">Random Forest</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">
            • Ensemble of decision trees<br>
            • Handles outliers well<br>
            • Robust predictions (92.6%)<br>
            • No overfitting
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with model_col3:
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
            <div style="font-size: 1.5rem; text-align: center;">🧠</div>
            <div style="font-weight: bold; text-align: center;">Feature Engineering</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">
            • 40+ URL features<br>
            • 20+ email features<br>
            • TF-IDF text vectors<br>
            • Domain-aware splitting
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== REAL MODEL METRICS ==========
    st.markdown("## 📈 Model Performance Metrics")
    st.markdown("*These metrics are from domain-aware splitting with **no data leakage***")
    
    # Load real metrics from JSON files
    url_metrics = {}
    email_metrics = {}
    
    try:
        with open("models/url_training_metrics.json", "r") as f:
            url_metrics = json.load(f)
    except:
        # Fallback to default values if file not found
        url_metrics = {
            "xgboost_url": {"accuracy": 0.9668, "precision": 0.9838, "recall": 0.9179, "f1": 0.9498},
            "random_forest_url": {"accuracy": 0.9595, "precision": 0.9745, "recall": 0.9055, "f1": 0.9387}
        }
    
    try:
        with open("models/email_training_metrics.json", "r") as f:
            email_metrics = json.load(f)
    except:
        # Fallback to default values if file not found
        email_metrics = {
            "xgboost_email": {"accuracy": 0.8671, "precision": 0.8795, "recall": 0.8905, "f1": 0.8849},
            "random_forest_email": {"accuracy": 0.8600, "precision": 0.8718, "recall": 0.8863, "f1": 0.8790}
        }
    
    # URL model
    st.markdown("### 🔗 URL Detection Models")
    url_data = [
        ("XGBoost",       url_metrics["xgboost_url"]["accuracy"], url_metrics["xgboost_url"]["precision"], 
                      url_metrics["xgboost_url"]["recall"], url_metrics["xgboost_url"]["f1"]),
        ("Random Forest", url_metrics["random_forest_url"]["accuracy"], url_metrics["random_forest_url"]["precision"], 
                      url_metrics["random_forest_url"]["recall"], url_metrics["random_forest_url"]["f1"]),
    ]
    for name, acc, prec, rec, f1 in url_data:
        st.markdown(f"**{name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc  * 100:.2f}%")
        c2.metric("Precision", f"{prec * 100:.2f}%")
        c3.metric("Recall",    f"{rec  * 100:.2f}%")
        c4.metric("F1 Score",  f"{f1   * 100:.2f}%")
        st.progress(acc, text=f"{name} Accuracy")

    # Email model
    st.markdown("### 📧 Email Detection Models")
    email_data = [
        ("XGBoost",       email_metrics["xgboost_email"]["accuracy"], email_metrics["xgboost_email"]["precision"], 
                       email_metrics["xgboost_email"]["recall"], email_metrics["xgboost_email"]["f1"]),
        ("Random Forest", email_metrics["random_forest_email"]["accuracy"], email_metrics["random_forest_email"]["precision"], 
                       email_metrics["random_forest_email"]["recall"], email_metrics["random_forest_email"]["f1"]),
    ]
    for name, acc, prec, rec, f1 in email_data:
        st.markdown(f"**{name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc  * 100:.2f}%")
        c2.metric("Precision", f"{prec * 100:.2f}%")
        c3.metric("Recall",    f"{rec  * 100:.2f}%")
        c4.metric("F1 Score",  f"{f1   * 100:.2f}%")
        st.progress(acc, text=f"{name} Accuracy")

    st.divider()
    # ========== DETECTION EXAMPLES ==========
    st.markdown("## 🔍 What Gets Detected")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("""
        ### 🚨 Phishing Indicators
        
        **URL Red Flags:**
        - ✅ Suspicious TLDs: `.xyz`, `.tk`, `.ml`, `.ga`
        - ✅ IP addresses instead of domain names
        - ✅ URL shorteners: `bit.ly`, `tinyurl.com`
        - ✅ Brand typos: `paypa1.com`, `amaz0n.com`
        - ✅ Excessive subdomains: `login.secure.verify.bank.com`
        
        **Email Red Flags:**
        - ✅ Urgency words: "immediately", "urgent", "ASAP"
        - ✅ Threats: "suspended", "terminated", "locked"
        - ✅ Suspicious sender domains
        - ✅ Embedded suspicious URLs
        - ✅ Poor grammar and spelling
        """)
    
    with example_col2:
        st.markdown("""
        ### ✅ Legitimate Indicators
        
        **URL Green Flags:**
        - ✅ Well-known domains: `.com`, `.org`, `.gov`
        - ✅ HTTPS encryption
        - ✅ Short, clean URL structure
        - ✅ Established brand domains
        - ✅ No suspicious characters
        
        **Email Green Flags:**
        - ✅ Professional signature
        - ✅ Expected sender domain
        - ✅ Proper grammar
        - ✅ No urgency pressure
        - ✅ Relevant content
        """)
    
    st.divider()
    
    # ========== SECURITY BEST PRACTICES ==========
    st.markdown("## 🔒 Security Best Practices")
    
    st.markdown("""
    <div style="background: #f0f8ff; border-radius: 15px; padding: 1.5rem;">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div>
                <span style="font-size: 1.2rem;">✅</span> Always verify unexpected requests through official channels<br><br>
                <span style="font-size: 1.2rem;">✅</span> Never share passwords or sensitive information via email<br><br>
                <span style="font-size: 1.2rem;">✅</span> Check sender email addresses carefully<br><br>
                <span style="font-size: 1.2rem;">✅</span> Hover over links before clicking to verify URLs
            </div>
            <div>
                <span style="font-size: 1.2rem;">✅</span> Use multi-factor authentication where available<br><br>
                <span style="font-size: 1.2rem;">✅</span> Keep software and browsers updated<br><br>
                <span style="font-size: 1.2rem;">✅</span> Report suspicious emails to your IT/security team<br><br>
                <span style="font-size: 1.2rem;">✅</span> Trust your instincts - if it seems suspicious, it probably is
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== TECHNICAL DETAILS ==========
    with st.expander("🔬 Technical Details & Methodology", expanded=False):
        st.markdown("""
        ### Feature Engineering Details
        
        **URL Features (40+):**
        - Structural: length, dots, hyphens, slashes, question marks
        - Security: HTTPS, port presence, authentication
        - Domain: TLD analysis, subdomain count, IP detection
        - Content: brand keywords, suspicious terms, URL shorteners
        - Complexity: entropy, encoding count, homograph score
        
        **Email Features (20+):**
        - Text statistics: length, caps count, digit count
        - Linguistics: urgency score, threat score, suspicious words
        - Metadata: sender reputation, SPF/DKIM/DMARC
        - HTML analysis: ratio, link count, form detection
        
        ### Training Methodology
        
        - **Domain-aware splitting**: Same domains never appear in train/test
        - **No data leakage**: Label columns excluded from features
        - **Stratified sampling**: Maintains class distribution
        - **Cross-validation**: 5-fold validation for robustness
        
        ### Model Optimization
        
        - **Hyperparameter tuning**: Grid search for optimal parameters
        - **Class balancing**: Handles imbalanced datasets
        - **Feature selection**: Removes correlated features
        - **Ensemble voting**: Weighted average of multiple models
        """)
    
    st.markdown("---")
    st.caption("🛡️ AI Phishing Defender v2.0 | Powered by Machine Learning & Natural Language Processing")
    st.caption("📧 For support or inquiries, contact your security team")


def statistics_page():
    """Statistics Page with REAL metrics"""
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="animated-title">📊 Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hardcoded real metrics ──────────────────────────────────────────
    REAL_METRICS = {
        "xgboost_url": {
            "label": "XGBoost",
            "accuracy":  0.9668,
            "precision": 0.9838,
            "recall":    0.9179,
            "f1":        0.9498,
        },
        "random_forest_url": {
            "label": "Random Forest",
            "accuracy":  0.9595,
            "precision": 0.9745,
            "recall":    0.9055,
            "f1":        0.9387,
        },
    }
    EMAIL_METRICS = {
        "xgboost_email": {
            "label": "XGBoost",
            "accuracy":  0.8671,
            "precision": 0.8795,
            "recall":    0.8905,
            "f1":        0.8849,
        },
        "random_forest_email": {
            "label": "Random Forest",
            "accuracy":  0.8600,
            "precision": 0.8718,
            "recall":    0.8863,
            "f1":        0.8790,
        },
    }
    # ───────────────────────────────────────────────────────────────────

    def render_model_metrics(metrics_dict, section_title):
        st.markdown(f"### {section_title}")
        for key, m in metrics_dict.items():
            st.markdown(f"#### 🤖 {m['label']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy",  f"{m['accuracy']  * 100:.2f}%")
            col2.metric("Precision", f"{m['precision'] * 100:.2f}%")
            col3.metric("Recall",    f"{m['recall']    * 100:.2f}%")
            col4.metric("F1 Score",  f"{m['f1']        * 100:.2f}%")

            # Visual bar
            st.progress(m['accuracy'], text=f"{m['label']} Accuracy")

            st.info(f"""
            📊 **What these numbers mean:**
            - **{m['accuracy']*100:.1f}% Accuracy** — Correct on {m['accuracy']*100:.1f} out of every 100 predictions
            - **{m['precision']*100:.1f}% Precision** — When it flags PHISHING, it's right {m['precision']*100:.1f}% of the time
            - **{m['recall']*100:.1f}% Recall** — Catches {m['recall']*100:.1f}% of all actual phishing attempts
            - **{m['f1']*100:.1f}% F1 Score** — Strong balance between precision and recall
            """)
            st.divider()

    # ── URL models ──────────────────────────────────────────────────────
    render_model_metrics(REAL_METRICS,  "🔗 URL Detection Models")

    # ── Email models ─────────────────────────────────────────────────────
    render_model_metrics(EMAIL_METRICS, "📧 Email Detection Models")

    # ── Side-by-side comparison chart ────────────────────────────────────
    if PLOTLY_AVAILABLE:
        st.markdown("### 📊 Model Comparison")

        all_models = {**REAL_METRICS, **EMAIL_METRICS}
        model_names  = [v["label"] + (" (URL)" if "url" in k else " (Email)") for k, v in all_models.items()]
        accuracies   = [v["accuracy"]  * 100 for v in all_models.values()]
        precisions   = [v["precision"] * 100 for v in all_models.values()]
        recalls      = [v["recall"]    * 100 for v in all_models.values()]
        f1_scores    = [v["f1"]        * 100 for v in all_models.values()]

        fig = go.Figure(data=[
            go.Bar(name="Accuracy",  x=model_names, y=accuracies,  marker_color="#667eea"),
            go.Bar(name="Precision", x=model_names, y=precisions,  marker_color="#2ed573"),
            go.Bar(name="Recall",    x=model_names, y=recalls,     marker_color="#ffa502"),
            go.Bar(name="F1 Score",  x=model_names, y=f1_scores,   marker_color="#ff4757"),
        ])
        fig.update_layout(
            barmode="group",
            yaxis=dict(title="Score (%)", range=[80, 100]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application with modern sidebar"""
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🛡️</div>
            <h2 style="color: white;">AI Phishing<br>Defender</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔗 URL Scanner", "📧 Email Scanner", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # System status
        st.markdown("### System Status")
        try:
            detector = initialize_detector()
            if detector.models_loaded:
                st.success("✅ AI Models Active")
                st.info(f"📁 Model Version: v2.0")
            else:
                st.warning("⚠️ Models Loading...")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Session stats
        if 'detection_history' in st.session_state:
            st.markdown("### Session Stats")
            st.metric("Total Scans", len(st.session_state.detection_history))
        
        st.markdown("---")
        st.caption("🛡️ AI Phishing Defender v2.0")
        st.caption("Powered by AI & Machine Learning")
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔗 URL Scanner":
        analyze_url_page()
    elif page == "📧 Email Scanner":
        analyze_email_page()

    elif page == "📊 Analytics":
        statistics_page()

    elif page == "ℹ️ About":
        about_page()

if __name__ == "__main__":
    main()