"""
Phishing Detector Module - Updated with IEEE and Organization Recognition
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import logging

from config import (
    MODEL_CONFIG, DETECTION_THRESHOLDS, SUSPICIOUS_WORDS,
    THREAT_KEYWORDS, SUSPICIOUS_TLDS, URL_ENTROPY_THRESHOLD
)
from feature_engineering import CombinedFeatureExtractor, EmailFeatureExtractor, URLFeatureExtractor
from models import EmailClassifier, EnsemblePhishingDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDetector:
    """
    Main phishing detection engine that combines multiple models
    for accurate real-time detection
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the phishing detector
        """
        self.model_path = model_path or 'models'
        self.feature_extractor = CombinedFeatureExtractor()
        self.url_extractor = URLFeatureExtractor()
        self.email_extractor = EmailFeatureExtractor()
        
        self.xgboost_model = None
        self.random_forest_model = None
        self.ensemble_model = None
        self.bert_model = None
        self.models_loaded = False
        
        # LEGITIMATE PROFESSIONAL SIGNATURE PATTERNS
        self.legitimate_signatures = [
            'ph.d', 'phd', 'professor', 'assistant professor', 'associate professor',
            'dr.', 'doctor', 'research scholar', 'dean', 'director',
            'national institute of technology', 'university', 'college',
            'assistant professor gr', 'department of', 'head of department',
            'ac.in', '.edu', '.ac.uk', '.edu.in',
            'ieee', 'acm', 'springer', 'elsevier', 'wiley', 'nature'
        ]
        
        # LEGITIMATE PROFESSIONAL PHRASES
        self.legitimate_phrases = [
            'best regards', 'sincerely', 'thanks', 'thank you',
            'looking forward', 'appreciate', 'warm regards',
            'kind regards', 'yours sincerely', 'yours faithfully',
            'not required', 'as requested', 'per our conversation',
            'ieee account', 'your ieee account', 'ieee services',
            'membership', 'subscription', 'publication', 'journal'
        ]
        
        # LEGITIMATE EMAIL DOMAINS (never flag these as suspicious)
        self.legitimate_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            '.ac.in', '.edu', '.gov', '.org', '.mil',
            'nits.ac.in', 'iit.ac.in', 'bits-pilani.ac.in',
            'ieee.org', 'ieee.com', 'ieee.net', 'ieeexplore.ieee.org',
            'emails.ieee.org', 'em3009.ieee.org',
            'acm.org', 'springer.com', 'elsevier.com', 'wiley.com',
            'nature.com', 'science.org', 'sciencedirect.com',
            'arxiv.org', 'researchgate.net', 'scholar.google.com',
            'orcid.org', 'crossref.org', 'scopus.com',
            'aaas.org', 'aps.org', 'acs.org', 'ama-assn.org', 'asm.org', 'asme.org',
            'conf.com', 'conference.org', 'proceedings.com',
            'iitb.ac.in', 'iitd.ac.in', 'iitm.ac.in', 'iitk.ac.in',
            'iitkgp.ac.in', 'iitr.ac.in', 'iitg.ac.in', 'iith.ac.in',
            'nits.ac.in', 'nitk.ac.in', 'nitw.ac.in', 'nitc.ac.in',
            'bits-pilani.ac.in', 'iiit.ac.in', 'isical.ac.in'
        ]
        
        # Brand patterns for impersonation detection (EXPANDED)
        self.brand_patterns = {
            'Google': ['google', 'g00gle', 'go0gle', 'googl', 'gogle'],
            'PayPal': ['paypal', 'pay-pal', 'paypa1', 'paypai', 'paypaI'],
            'Microsoft': ['microsoft', 'micros0ft', 'microsfot', 'micr0soft'],
            'Apple': ['apple', 'app1e', 'aple', 'appie'],
            'Amazon': ['amazon', 'amaz0n', 'amzon', 'amazzon'],
            'Facebook': ['facebook', 'faceb00k', 'facebok', 'facbook'],
            'DataHub': ['datahub', 'data-hub', 'data_hub'],
            'Dropbox': ['dropbox', 'dropboox', 'drop-box'],
            'LinkedIn': ['linkedin', 'linkedln', 'linkdin'],
            'Netflix': ['netflix', 'netflx', 'netflixx'],
            'Kraken': ['kraken', 'kraekken', 'krakin', 'krakken', 'krakn'],
            'MetaMask': ['metamask', 'metamsk', 'metamas', 'metamaks', 'metamak'],
            'Coinbase': ['coinbase', 'coinbass', 'coinbas', 'coinbaze'],
            'Binance': ['binance', 'binanc', 'binans', 'binancee'],
            'Trust Wallet': ['trustwallet', 'trust-wallet', 'truste'],
            'Ledger': ['ledger', 'ledgr', 'ledgar', 'ledjer', 'ledgher'],
            'Robinhood': ['robinhood', 'robinhod', 'robinhoodd'],
            'Godaddy': ['godaddy', 'godaddysites', 'godady', 'go-daddy', 'goddady']
        }
        
        # Legitimate domains for each brand (EXPANDED)
        self.legit_domains = {
            'Google': ['google.com', 'google.co', 'gmail.com', 'googleapis.com'],
            'PayPal': ['paypal.com', 'paypal.cn'],
            'Microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com', 'msn.com'],
            'Apple': ['apple.com', 'icloud.com', 'appleid.apple.com'],
            'Amazon': ['amazon.com', 'amazon.co', 'amazon.de', 'amazon.fr', 'amazon.co.uk'],
            'Facebook': ['facebook.com', 'fb.com', 'messenger.com', 'instagram.com'],
            'DataHub': ['datahub.org', 'datahub.io', 'datahub.com'],
            'Dropbox': ['dropbox.com'],
            'LinkedIn': ['linkedin.com'],
            'Netflix': ['netflix.com'],
            'Kraken': ['kraken.com', 'kraken.io'],
            'MetaMask': ['metamask.io', 'metamask.com'],
            'Coinbase': ['coinbase.com', 'coinbase.io'],
            'Binance': ['binance.com', 'binance.us'],
            'Trust Wallet': ['trustwallet.com'],
            'Ledger': ['ledger.com', 'ledgerwallet.com'],
            'Robinhood': ['robinhood.com'],
            'Godaddy': ['godaddy.com']
        }
        
        # Misleading keywords
        self.misleading_keywords = [
            'verify-now', 'confirm-identity', 'account-suspended', 
            'unusual-activity', 'secure-login', 'update-billing'
        ]
        
        # Risk indicators
        self.risk_indicators = []
        
        # Auto-load models on initialization
        self.load_models()
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result dictionary"""
        return {
            'error': error_message,
            'risk_score': 0.0,
            'risk_level': 'ERROR',
            'risk_factors': [error_message],
            'ml_prediction': None,
            'ml_confidence': None,
            'features': {}
        }
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            ensemble_path = f"{self.model_path}/ensemble"
            if Path(ensemble_path).exists():
                self.ensemble_model = EnsemblePhishingDetector()
                self.ensemble_model.load(ensemble_path)
                self.models_loaded = True
                logger.info("Ensemble model loaded successfully")
                return True
            
            xgb_path = MODEL_CONFIG['xgboost_path']
            rf_path = MODEL_CONFIG['random_forest_path']
            
            if Path(xgb_path).exists():
                self.xgboost_model = EmailClassifier(use_xgboost=True)
                self.xgboost_model.load(xgb_path)
                logger.info("XGBoost model loaded successfully")
            
            if Path(rf_path).exists():
                self.random_forest_model = EmailClassifier(use_xgboost=False)
                self.random_forest_model.load(rf_path)
                logger.info("Random Forest model loaded successfully")
            
            # Load standalone BERT model if available
            bert_path = f"{self.model_path}/bert_email_model"
            if Path(bert_path).exists():
                try:
                    from models import BERTURLClassifier
                    self.bert_model = BERTURLClassifier()
                    self.bert_model.load(bert_path)
                    logger.info("BERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"BERT model could not be loaded: {e}")
                    self.bert_model = None

            self.models_loaded = any([self.xgboost_model, self.random_forest_model, self.bert_model])
            if self.models_loaded:
                logger.info("Models loaded successfully")
            else:
                logger.warning("No models found. Run train.py first.")
            
            return self.models_loaded
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _is_legitimate_professional_email(self, subject: str, body: str, sender: str) -> tuple:
        """Check if email appears to be legitimate professional/academic communication"""
        full_text = f"{subject} {body}".lower()
        sender_lower = sender.lower()
        
        legitimate_orgs = [
            'ieee', 'acm', 'springer', 'elsevier', 'wiley', 'nature',
            'science', 'arxiv', 'researchgate', 'google scholar', 'orcid'
        ]
        
        for org in legitimate_orgs:
            if org in full_text or org in sender_lower:
                if not self._has_phishing_indicators(full_text):
                    return (True, f"Legitimate communication from {org.upper()}")
        
        signature_score = 0
        for pattern in self.legitimate_signatures:
            if pattern.lower() in full_text:
                signature_score += 1
        
        phrase_score = 0
        for phrase in self.legitimate_phrases:
            if phrase.lower() in full_text:
                phrase_score += 1
        
        domain_legitimate = False
        for domain in self.legitimate_domains:
            if domain in sender_lower:
                domain_legitimate = True
                break
        
        has_title = any(title in full_text for title in ['dr.', 'professor', 'ph.d', 'phd', 'ieee'])
        has_academic_domain = any(domain in sender_lower for domain in ['.ac.in', '.edu', '.ac.uk', 'ieee.org'])
        has_professional_signature = signature_score >= 2 or (has_title and signature_score >= 1)
        
        if has_academic_domain and has_professional_signature:
            return (True, "Academic institution email with professional signature")
        
        if domain_legitimate and phrase_score >= 1:
            return (True, "Legitimate email domain with professional closing")
        
        if has_professional_signature and phrase_score >= 1:
            return (True, "Professional signature and closing detected")
        
        if len(full_text) < 500 and phrase_score >= 2 and not self._has_phishing_indicators(full_text):
            return (True, "Short professional communication")
        
        if 'ieee' in sender_lower or 'ieee' in full_text:
            if not self._has_phishing_indicators(full_text):
                return (True, "IEEE official communication")
        
        return (False, "")
    
    def _has_phishing_indicators(self, text: str) -> bool:
        """Check for strong phishing indicators"""
        text_lower = text.lower()
        
        strong_indicators = [
            'verify your account immediately', 'confirm your identity now',
            'your account has been suspended', 'click here to verify immediately',
            'update your payment information now', 'your account will be permanently closed',
            'unusual activity detected on your account', 'your account has been limited',
            'verify your identity to restore', 'wire transfer', 'western union', 'money gram',
            'social security number', 'ssn', 'credit card number',
            'bank account details required', 'dear winner', 'you have won',
            'claim your prize now', 'inheritance', 'lottery winner'
        ]
        
        for indicator in strong_indicators:
            if indicator in text_lower:
                return True
        
        url_patterns = [
            r'http://[^\s]*\.tk[/]', r'http://[^\s]*\.ml[/]', r'http://[^\s]*\.ga[/]',
            r'http://[^\s]*\.cf[/]', r'http://[^\s]*\.gq[/]', r'http://[^\s]*\.xyz[/]'
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _check_brand_impersonation(self, url: str) -> str:
        """Check if URL attempts to impersonate a known brand/service"""
        url_lower = url.lower()
        
        for brand, patterns in self.brand_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    legit_domains = self.legit_domains.get(brand, [])
                    for legit in legit_domains:
                        if legit in url_lower:
                            return ""
                    return brand
        return ""
    
    def _check_suspicious_url_patterns(self, url: str) -> List[str]:
        """Check for suspicious URL patterns"""
        url_lower = url.lower()
        suspicious = []
        
        if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            suspicious.append("IP address used instead of domain name")
        
        subdomain_count = url_lower.count('.') - url_lower.count('://') - 1
        if subdomain_count > 3:
            suspicious.append(f"Excessive subdomains ({subdomain_count})")
        
        for tld in SUSPICIOUS_TLDS:
            if tld in url_lower:
                suspicious.append(f"Suspicious TLD ({tld})")
                break
        
        if '@' in url and 'mailto:' not in url:
            suspicious.append("@ symbol in URL - possible credential theft")
        
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co']
        for shortener in shorteners:
            if shortener in url_lower:
                suspicious.append(f"URL shortener ({shortener}) - hides destination")
                break
        
        return suspicious
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze a URL for phishing indicators"""
        if not url or not isinstance(url, str):
            return self._error_result("Invalid URL provided")
        
        try:
            features = self.url_extractor.extract_features(url)
        except Exception as e:
            logger.error(f"Error extracting URL features: {e}")
            return self._error_result(f"Error analyzing URL: {str(e)}")
        
        risk_score = 0.0
        risk_factors = []
        url_lower = url.lower()
        
        # Brand impersonation
        impersonated_brand = self._check_brand_impersonation(url)
        if impersonated_brand:
            risk_score += 0.55
            risk_factors.append(f"🚨 Brand impersonation: pretending to be '{impersonated_brand}'")
        
        # FREE HOSTING DETECTION
        free_hosting_sites = ['godaddysites.com', 'weebly.com', 'wixsite.com', 'pages.dev', 
                              'firebaseapp.com', 'netlify.app', 'vercel.app', 'github.io',
                              'gitlab.io', 'herokuapp.com', 'cloudfront.net']
        
        for hosting in free_hosting_sites:
            if hosting in url_lower:
                for brand in ['kraken', 'paypal', 'metamask', 'coinbase', 'binance', 
                             'trust', 'ledger', 'robinhood', 'amazon', 'microsoft', 'apple']:
                    if brand in url_lower:
                        risk_score += 0.45
                        risk_factors.append(f"🚨 Suspicious {brand.title()} lookalike on free hosting ({hosting})")
                        break
                try:
                    subdomain = url_lower.split('://')[1].split('.')[0]
                    if len(subdomain) > 12:
                        risk_score += 0.15
                        risk_factors.append(f"⚠️ Suspicious long subdomain on free hosting")
                    if any(c.isdigit() for c in subdomain) and len(subdomain) > 8:
                        risk_score += 0.10
                        risk_factors.append(f"⚠️ Numeric subdomain on free hosting")
                except:
                    pass
                break
        
        # Suspicious URL patterns
        suspicious_patterns = self._check_suspicious_url_patterns(url)
        for pattern in suspicious_patterns:
            risk_score += 0.25
            risk_factors.append(f"⚠️ {pattern}")
        
        # Missing HTTPS with sensitive keywords
        if not features.get('has_https', False):
            sensitive_keywords = ['login', 'signin', 'account', 'payment', 'bank', 'paypal']
            if any(keyword in url_lower for keyword in sensitive_keywords):
                risk_score += 0.25
                risk_factors.append("⚠️ No HTTPS encryption on login/payment page")
        
        # ML Model prediction
        ml_prediction = None
        ml_confidence = None
        
        if self.models_loaded and self.ensemble_model:
            try:
                feature_names = self.url_extractor.get_feature_names()
                feature_values = [features.get(name, 0.0) for name in feature_names]
                X = np.array([feature_values])
                
                proba = self.ensemble_model.predict_proba(X)
                ml_prediction = int(proba[0][1] >= 0.5)
                ml_confidence = float(proba[0][1])
                
                if ml_prediction == 1:
                    risk_score = max(risk_score, ml_confidence * 1.1)
                    risk_factors.append(f"🤖 ML Model: PHISHING detected ({ml_confidence:.1%} confidence)")
                else:
                    risk_score = min(risk_score, 0.2)
                    
            except Exception as e:
                logger.error(f"ML prediction error: {e}")
                # BERT prediction for URL text
        if self.bert_model and self.bert_model.is_fitted:
            try:
                bert_proba = self.bert_model.predict_proba([url])
                bert_confidence = float(bert_proba[0][1])
                bert_prediction = int(bert_confidence >= 0.5)

                if bert_prediction == 1:
                    # Blend BERT score with existing risk_score
                    risk_score = max(risk_score, bert_confidence * 0.9)
                    risk_factors.append(
                        f"🤖 BERT Model: PHISHING detected ({bert_confidence:.1%} confidence)"
                    )
                else:
                    # BERT says safe — dampen risk slightly but don't override hard rules
                    risk_score = risk_score * 0.85
                    risk_factors.append(
                        f"✅ BERT Model: Legitimate ({(1 - bert_confidence):.1%} confidence)"
                    )
            except Exception as e:
                logger.error(f"BERT URL prediction error: {e}")
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.70:
            risk_level = "HIGH"
        elif risk_score >= 0.45:
            risk_level = "MEDIUM"
        elif risk_score >= 0.20:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"
        
        return {
            'url': url,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors[:8],
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'features': features
        }
    
    def analyze_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an email for phishing indicators"""
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        sender = email_data.get('sender', '')
        full_text = f"{subject} {body}".lower()
        
        is_legitimate, legit_reason = self._is_legitimate_professional_email(subject, body, sender)
        is_ieee = 'ieee' in sender.lower() or 'ieee' in full_text
        
        if is_legitimate or is_ieee:
            reason = legit_reason if legit_reason else "IEEE or legitimate organization communication"
            return {
                'subject': subject,
                'sender': sender,
                'risk_score': 0.05,
                'risk_level': "SAFE",
                'risk_factors': [f"✅ {reason}"],
                'ml_prediction': 0,
                'ml_confidence': 0.0,
                'features': {}
            }
        
        features = self.feature_extractor.extract_from_email_with_urls(email_data)
        
        risk_score = 0.0
        risk_factors = []
        
        if self._has_phishing_indicators(full_text):
            risk_score += 0.50
            risk_factors.append("🚨 Strong phishing indicators detected")
        
        total_urls = features.get('total_urls_in_email', 0)
        suspicious_urls = features.get('suspicious_url_count', 0)
        
        if suspicious_urls > 0:
            risk_score += min(suspicious_urls * 0.35, 0.55)
            risk_factors.append(f"🚨 {suspicious_urls} suspicious URL(s) found")
        
        if sender and '@' in sender:
            sender_domain = sender.split('@')[1].lower()
            major_brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'bank', 'chase']
            mentioned_brands = [b for b in major_brands if b in full_text]
            
            if mentioned_brands:
                brand_domains = {
                    'paypal': 'paypal.com',
                    'amazon': 'amazon.com',
                    'microsoft': 'microsoft.com',
                    'apple': 'apple.com',
                    'google': 'google.com'
                }
                
                for brand in mentioned_brands:
                    expected_domain = brand_domains.get(brand, '')
                    if expected_domain and expected_domain not in sender_domain:
                        if sender_domain in ['gmail.com', 'yahoo.com', 'hotmail.com']:
                            risk_score += 0.40
                            risk_factors.append(f"⚠️ Claims to be from {brand.title()} but uses {sender_domain}")
        
        ml_prediction = None
        ml_confidence = None
        
        if self.models_loaded:
            try:
                feature_names = self.feature_extractor.get_all_feature_names()
                feature_values = [features.get(name, 0.0) for name in feature_names]
                X = np.array([feature_values])
                
                text = f"{subject} {body}"
                texts = [text]
                
                if self.ensemble_model:
                    proba = self.ensemble_model.predict_proba(X, texts)
                    ml_prediction = int(proba[0][1] >= 0.5)
                    ml_confidence = float(proba[0][1])
                elif self.xgboost_model:
                    proba = self.xgboost_model.predict_proba(X)
                    ml_prediction = int(proba[0][1] >= 0.5)
                    ml_confidence = float(proba[0][1])
                
                if ml_prediction == 1:
                    risk_score = max(risk_score, ml_confidence * 1.1)
                    risk_factors.append(f"🤖 ML Model: PHISHING detected ({ml_confidence:.1%} confidence)")
                else:
                    risk_score = risk_score * 0.4
                    
            except Exception as e:
                logger.error(f"ML prediction error: {e}")
        # BERT prediction for email text
        if self.bert_model and self.bert_model.is_fitted:
            try:
                email_text = f"{subject} {body}"
                bert_proba = self.bert_model.predict_proba([email_text])
                bert_confidence = float(bert_proba[0][1])
                bert_prediction = int(bert_confidence >= 0.5)

                if bert_prediction == 1:
                    # If BERT strongly agrees it is phishing, push the score up
                    if bert_confidence > 0.75:
                        risk_score = max(risk_score, bert_confidence * 0.95)
                    else:
                        risk_score = max(risk_score, bert_confidence * 0.7)
                    risk_factors.append(
                        f"🤖 BERT Model: PHISHING detected ({bert_confidence:.1%} confidence)"
                    )
                else:
                    # BERT says legitimate — reduce risk score but not eliminate it
                    risk_score = risk_score * 0.75
                    risk_factors.append(
                        f"✅ BERT Model: Legitimate ({(1 - bert_confidence):.1%} confidence)"
                    )
            except Exception as e:
                logger.error(f"BERT email prediction error: {e}")
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.65:
            risk_level = "HIGH"
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
        elif risk_score >= 0.15:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"
        
        if risk_level == "SAFE" and risk_score < 0.10:
            risk_factors.insert(0, "✅ Email appears legitimate")
        elif risk_level == "HIGH":
            risk_factors.insert(0, "🔴 HIGH RISK: Multiple phishing indicators detected")
        elif risk_level == "MEDIUM":
            risk_factors.insert(0, "🟠 MEDIUM RISK: Some suspicious elements detected")
        
        return {
            'subject': subject,
            'sender': sender,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors[:8],
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'features': features
        }
    
    def detect_phishing_url(self, url: str) -> Dict[str, Any]:
        """Quick URL phishing detection"""
        return self.analyze_url(url)
    
    def detect_phishing_email(self, subject: str, body: str, 
                              sender: str = None, 
                              headers: Dict = None) -> Dict[str, Any]:
        """Quick email phishing detection"""
        email_data = {
            'subject': subject,
            'body': body,
            'sender': sender or '',
            'headers': headers or {},
            'html_content': ''
        }
        return self.analyze_email(email_data)
    
    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Get security recommendations based on analysis results"""
        recommendations = []
        risk_level = analysis_result.get('risk_level', 'SAFE')
        
        if risk_level == "HIGH":
            recommendations.extend([
                "🚨 DO NOT click any links in this message",
                "🚨 DO NOT reply or provide any information",
                "Report this as phishing to your IT team",
                "Delete this message immediately",
                "If you already clicked a link, change your passwords now"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "⚠️ Exercise caution with this message",
                "⚠️ Verify the sender through another channel",
                "⚠️ Hover over links before clicking",
                "Consider reporting as suspicious"
            ])
        elif risk_level == "LOW":
            recommendations.extend([
                "ℹ️ Minor concerns detected",
                "ℹ️ Still verify unexpected requests",
                "ℹ️ Be cautious with any links"
            ])
        else:
            recommendations.extend([
                "✅ This message appears safe",
                "✅ Continue normal security practices"
            ])
        
        return recommendations
    
    def batch_analyze_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple URLs at once"""
        results = []
        for url in urls:
            try:
                results.append(self.analyze_url(url))
            except Exception as e:
                logger.error(f"Error analyzing URL {url}: {e}")
                results.append(self._error_result(f"Error: {str(e)}"))
        return results
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        cleaned_urls = []
        for url in urls:
            url = url.rstrip('.,;:!?)>')
            if url and len(url) > 10:
                cleaned_urls.append(url)
        return list(set(cleaned_urls))
    
    def analyze_email_with_url_extraction(self, subject: str, body: str,
                                          sender: str = None) -> Dict[str, Any]:
        """Analyze email and extract/analyze all embedded URLs"""
        email_result = self.detect_phishing_email(subject, body, sender)
        
        full_text = f"{subject} {body}"
        urls = self.extract_urls_from_text(full_text)
        
        url_analyses = []
        if urls:
            url_analyses = self.batch_analyze_urls(urls)
        
        combined_risk_score = email_result['risk_score']
        if url_analyses:
            max_url_risk = max(ua['risk_score'] for ua in url_analyses)
            combined_risk_score = max(combined_risk_score, max_url_risk)
        
        if combined_risk_score >= 0.65:
            combined_risk_level = "HIGH"
        elif combined_risk_score >= 0.40:
            combined_risk_level = "MEDIUM"
        elif combined_risk_score >= 0.15:
            combined_risk_level = "LOW"
        else:
            combined_risk_level = "SAFE"
        
        return {
            'email_analysis': email_result,
            'url_analyses': url_analyses,
            'urls_found': urls,
            'combined_risk_score': combined_risk_score,
            'combined_risk_level': combined_risk_level,
            'recommendations': self.get_recommendations({
                'risk_level': combined_risk_level
            })
        }


_detector_instance = None


def get_detector() -> PhishingDetector:
    """Get or create the singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PhishingDetector()
    return _detector_instance