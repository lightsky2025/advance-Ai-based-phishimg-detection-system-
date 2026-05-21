"""
Feature Engineering Module for Phishing Detection System
Handles email and URL feature extraction
"""

import re
import math
import string
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Any, Optional
import numpy as np

from config import (
    SUSPICIOUS_WORDS, THREAT_KEYWORDS, SUSPICIOUS_TLDS, 
    URL_SHORTENERS, FEATURE_CONFIG
)


class EmailFeatureExtractor:
    """Extract features from email content and metadata"""
    
    def __init__(self):
        self.feature_names = FEATURE_CONFIG["email_features"]
        
    def extract_features(self, email_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features from email data
        
        Args:
            email_data: Dictionary containing email fields:
                - subject: Email subject
                - body: Email body content
                - sender: Sender email address
                - headers: Email headers dict
                - html_content: HTML part of email (optional)
                
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Combine subject and body for text analysis
        text = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        html_content = email_data.get('html_content', '')
        
        # Basic text features
        features['total_length'] = len(text)
        features['num_caps'] = sum(1 for c in text if c.isupper())
        features['num_digits'] = sum(1 for c in text if c.isdigit())
        features['num_special_chars'] = sum(1 for c in text if c in string.punctuation)
        
        # URL and link analysis
        features['num_urls'] = len(re.findall(r'https?://\S+', text))
        features['num_emails'] = len(re.findall(r'\S+@\S+\.\S+', text))
        features['num_ips'] = len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text))
        features['num_links'] = len(re.findall(r'<a\s+.*?href.*?>', html_content, re.IGNORECASE))
        
        # Attachment detection
        features['num_attachments'] = email_data.get('num_attachments', 0)
        
        # Linguistic features
        features['suspicious_words_count'] = self._count_suspicious_words(text)
        features['urgency_score'] = self._calculate_urgency_score(text)
        features['threat_score'] = self._calculate_threat_score(text)
        
        # Sender reputation (simplified - would integrate with reputation API in production)
        features['sender_reputation'] = self._calculate_sender_reputation(
            email_data.get('sender', '')
        )
        
        # Authentication headers
        headers = email_data.get('headers', {})
        features['has_spf'] = 1 if any(k.lower().startswith('spf') for k in headers) else 0
        features['has_dkim'] = 1 if any(k.lower().startswith('dkim') for k in headers) else 0
        features['has_dmarc'] = 1 if 'dmarc' in str(headers).lower() else 0
        features['spf_result_match'] = self._check_spf_match(headers)
        
        # HTML analysis
        features['html_ratio'] = self._calculate_html_ratio(html_content, text)
        features['text_html_ratio'] = self._calculate_text_html_ratio(text, html_content)
        
        # Text complexity features
        features['avg_word_length'] = self._calculate_avg_word_length(text)
        features['unique_word_ratio'] = self._calculate_unique_word_ratio(text)
        
        return features
    
    def _count_suspicious_words(self, text: str) -> float:
        """Count suspicious/phishing-related words"""
        text_lower = text.lower()
        count = sum(1 for word in SUSPICIOUS_WORDS if word in text_lower)
        return count / len(SUSPICIOUS_WORDS)  # Normalize
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on language patterns"""
        urgency_patterns = [
            r'\b(urgent|immediately|asap|emergency|critical)\b',
            r'\b(act now|click here|verify now|confirm immediately)\b',
            r'\b(will expire|account suspended|terminate|deadline)\b',
            r'!!!+',  # Multiple exclamation marks
            r'\b([A-Z]{5,})\b'  # All caps words of length 5+
        ]
        
        score = 0
        text_lower = text.lower()
        for pattern in urgency_patterns:
            score += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Normalize to 0-1 range
        return min(score / 10, 1.0)
    
    def _calculate_threat_score(self, text: str) -> float:
        """Calculate threat score based on threatening language"""
        text_lower = text.lower()
        count = sum(1 for keyword in THREAT_KEYWORDS if keyword in text_lower)
        return min(count / len(THREAT_KEYWORDS), 1.0)
    
    def _calculate_sender_reputation(self, sender: str) -> float:
        """
        Calculate sender reputation score
        In production, this would integrate with reputation databases
        """
        if not sender:
            return 0.5  # Neutral for unknown
        
        # Check for suspicious patterns
        reputation = 0.5  # Start neutral
        
        # Free email domains often used in phishing
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if any(domain in sender.lower() for domain in free_domains):
            reputation -= 0.1
        
        # Check for suspicious characters
        if re.search(r'[0-9]{3,}', sender.split('@')[0]):
            reputation -= 0.1
        
        # Check for suspicious TLDs
        if any(tld in sender for tld in SUSPICIOUS_TLDS):
            reputation -= 0.2
        
        return max(0, min(reputation, 1.0))
    
    def _check_spf_match(self, headers: Dict) -> float:
        """Check if SPF authentication passed"""
        for key, value in headers.items():
            if 'spf' in key.lower():
                if 'pass' in str(value).lower():
                    return 1.0
                elif 'fail' in str(value).lower():
                    return 0.0
        return 0.5  # Unknown
    
    def _calculate_html_ratio(self, html_content: str, text: str) -> float:
        """Calculate ratio of HTML content to total content"""
        if not html_content:
            return 0.0
        total = len(html_content) + len(text)
        if total == 0:
            return 0.0
        return len(html_content) / total
    
    def _calculate_text_html_ratio(self, text: str, html_content: str) -> float:
        """Calculate ratio of plain text to HTML content"""
        if not html_content:
            return 1.0
        if len(html_content) == 0:
            return 1.0
        return len(text) / len(html_content)
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length"""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_unique_word_ratio(self, text: str) -> float:
        """Calculate ratio of unique words to total words"""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection - ENHANCED VERSION"""
    
    def __init__(self):
        self.feature_names = FEATURE_CONFIG["url_features"]
        
    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extract enhanced features from a URL
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Basic URL features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_at_symbols'] = url.count('@')
        features['num_question_marks'] = url.count('?')
        features['num_percent'] = url.count('%')
        features['num_equals'] = url.count('=')
        features['num_slashes'] = url.count('/')
        
        # Protocol analysis
        features['has_https'] = 1 if url.startswith('https://') else 0
        
        # Parse URL for deeper analysis
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc
            path = parsed.path
            query = parsed.query
        except:
            hostname = ''
            path = ''
            query = ''
        
        # TLD analysis
        features['has_suspicious_tld'] = 1 if self._has_suspicious_tld(url) else 0
        
        # IP address detection
        features['has_ip_address'] = 1 if self._is_ip_address(hostname) else 0
        
        # URL shortening detection
        features['has_shortening_service'] = 1 if self._is_url_shortener(url) else 0
        
        # Subdomain analysis
        features['num_subdomains'] = self._count_subdomains(hostname)
        
        # Entropy calculation (higher entropy may indicate obfuscation)
        features['url_entropy'] = self._calculate_entropy(url)
        
        # Port detection
        features['has_port'] = 1 if ':' in hostname else 0
        
        # Authentication in URL
        features['has_authentication'] = 1 if '@' in parsed.netloc else 0
        
        # Path and query analysis
        features['path_length'] = len(path)
        features['query_length'] = len(query)
        
        # Character ratios
        features['digit_url_ratio'] = sum(1 for c in url if c.isdigit()) / max(len(url), 1)
        features['special_char_ratio'] = sum(1 for c in url if c in string.punctuation) / max(len(url), 1)
        
        # ========== NEW FEATURES - ADD THESE ==========
        
        # 1. Number of redirects (detects URL shorteners and redirect chains)
        features['num_redirects'] = url.count('//') - 1
        if features['num_redirects'] < 0:
            features['num_redirects'] = 0
        
        # 2. Tracking parameters detection (phishers use these to look legitimate)
        tracking_params = ['utm_', 'fbclid', 'gclid', 'ref=', 'source=', 'campaign=', 'medium=']
        features['has_tracking'] = 1 if any(param in url.lower() for param in tracking_params) else 0
        
        # 3. Current year in URL (phishing sites often use current year)
        import datetime
        current_year = datetime.datetime.now().year
        features['has_current_year'] = 1 if str(current_year) in url else 0
        
        # Also check for previous year (2025, 2024, etc. - common in phishing)
        features['has_any_year'] = 1 if any(str(year) in url for year in range(2020, current_year + 1)) else 0
        
        # 4. Path depth (deep nesting is suspicious)
        path_parts = [p for p in path.split('/') if p]
        features['path_depth'] = len(path_parts)
        
        # 5. Digit ratio in hostname (phishing domains often have many digits)
        hostname_clean = re.sub(r'^www\.', '', hostname)
        hostname_clean = hostname_clean.split(':')[0]  # Remove port
        digits_in_host = sum(c.isdigit() for c in hostname_clean)
        features['digit_ratio_host'] = digits_in_host / max(len(hostname_clean), 1)
        
        # 6. Number of hyphens in hostname (fake domains use hyphens)
        features['hyphens_in_host'] = hostname_clean.count('-')
        
        # 7. Brand keyword detection (paypal, amazon, etc. in URL)
        brand_keywords = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 
                          'facebook', 'netflix', 'kraken', 'coinbase', 'binance']
        features['brand_keyword_count'] = sum(1 for brand in brand_keywords if brand in url.lower())
        
        # 8. Suspicious keyword detection
        suspicious_keywords = ['verify', 'secure', 'login', 'signin', 'account', 
                               'update', 'confirm', 'alert', 'suspended', 'unusual']
        features['suspicious_keyword_count'] = sum(1 for kw in suspicious_keywords if kw in url.lower())
        
        # 9. URL has uppercase letters (legitimate rarely use all caps in paths)
        features['has_uppercase'] = 1 if any(c.isupper() for c in url) else 0
        
        # 10. Number of parameters in query string
        features['num_query_params'] = query.count('&') + 1 if query else 0
        
        # 11. Check for common TLDs vs suspicious TLDs
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov']
        features['is_common_tld'] = 1 if any(tld in url.lower() for tld in common_tlds) else 0
        
        # 12. Check if URL contains @ symbol (credential theft)
        features['has_at_symbol'] = 1 if '@' in url and 'mailto:' not in url else 0
        
        # 13. Check for double dots (directory traversal attempt)
        features['has_double_dot'] = 1 if '..' in url else 0
        
        # 14. Check for excessive encoding (% signs)
        features['has_encoding'] = 1 if '%' in url else 0
        features['encoding_count'] = url.count('%')
        
        # 15. Check for multiple subdomains (legitimate rarely have > 2)
        subdomain_count = features.get('num_subdomains', 0)
        features['has_excessive_subdomains'] = 1 if subdomain_count > 2 else 0
        
        # ========== ADVANCED PATTERN DETECTION ==========
        
        # 16. Detect homograph attacks (similar looking characters)
        homograph_chars = ['0', '1', 'l', 'I', 'O', 'o', 'rn', 'vv', 'vv']
        homograph_count = 0
        for char in homograph_chars:
            if char in url.lower():
                homograph_count += 1
        features['homograph_score'] = min(homograph_count / 5, 1.0)
        
        # 17. Check for brand typosquatting
        brand_typos = {
            'paypa1': 'paypal', 'g00gle': 'google', 'amaz0n': 'amazon',
            'micr0soft': 'microsoft', 'faceb00k': 'facebook', 'kraekken': 'kraken'
        }
        features['has_typosquatting'] = 0
        for typo, brand in brand_typos.items():
            if typo in url.lower():
                features['has_typosquatting'] = 1
                break
        
        # 18. URL complexity score (combination of entropy and length)
        features['url_complexity'] = min((features['url_entropy'] * len(url) / 50) / 10, 1.0)
        
        return features
    
    def _has_suspicious_tld(self, url: str) -> bool:
        """Check if URL has a suspicious TLD"""
        url_lower = url.lower()
        return any(tld in url_lower for tld in SUSPICIOUS_TLDS)
    
    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address"""
        host = hostname.split(':')[0]
        pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(pattern, host))
    
    def _is_url_shortener(self, url: str) -> bool:
        """Check if URL is from a URL shortening service"""
        url_lower = url.lower()
        return any(shortener in url_lower for shortener in URL_SHORTENERS)
    
    def _count_subdomains(self, hostname: str) -> int:
        """Count number of subdomains"""
        host = hostname.split(':')[0]
        parts = host.split('.')
        if len(parts) <= 2:
            return 0
        return max(0, len(parts) - 2)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the URL"""
        if not text:
            return 0.0
        
        prob = {}
        for char in text:
            prob[char] = prob.get(char, 0) + 1
        
        length = len(text)
        entropy = 0.0
        for count in prob.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names

class CombinedFeatureExtractor:
    """Combined extractor for both email and URL features"""
    
    def __init__(self):
        self.email_extractor = EmailFeatureExtractor()
        self.url_extractor = URLFeatureExtractor()
        
    def extract_email_features(self, email_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from email data"""
        return self.email_extractor.extract_features(email_data)
    
    def extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract features from URL"""
        return self.url_extractor.extract_features(url)
    
    def extract_from_email_with_urls(self, email_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract combined features from email including embedded URLs
        """
        # Get email features
        features = self.email_extractor.extract_features(email_data)
        
        # Extract URLs from email body
        body = email_data.get('body', '')
        urls = re.findall(r'https?://\S+', body)
        
        # Add URL-based features
        if urls:
            url_features_list = [self.url_extractor.extract_features(url) for url in urls]
            
            # Aggregate URL features (mean)
            for key in self.url_extractor.get_feature_names():
                values = [uf[key] for uf in url_features_list]
                features[f'url_{key}'] = float(np.mean(values)) if values else 0.0
            
            # Add URL count and risk indicators
            features['total_urls_in_email'] = len(urls)
            features['suspicious_url_count'] = sum(
                1 for uf in url_features_list 
                if uf.get('has_suspicious_tld', 0) or uf.get('has_ip_address', 0)
            )
        else:
            # Add zero values for URL features if no URLs found
            for key in self.url_extractor.get_feature_names():
                features[f'url_{key}'] = 0.0
            features['total_urls_in_email'] = 0
            features['suspicious_url_count'] = 0
        
        return features
    
    def get_all_feature_names(self) -> List[str]:
        """Get all possible feature names"""
        email_features = self.email_extractor.get_feature_names()
        url_features = [f'url_{f}' for f in self.url_extractor.get_feature_names()]
        extra_features = ['total_urls_in_email', 'suspicious_url_count']
        return email_features + url_features + extra_features