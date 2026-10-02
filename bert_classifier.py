"""
bert_classifier.py
==================
Deep Learning Phishing Email Detection using BERT / DistilBERT Transformers.
Provides state-of-the-art semantic comprehension, contextual attention,
and robust detection of obfuscated and subtle phishing techniques.
"""

import os
import sys
import re
import torch
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default pre-trained high-accuracy phishing detection model
DEFAULT_PRETRAINED_MODEL = 'cybersectony/phishing-email-detection-distilbert_v2.4.1'
LOCAL_BERT_PATH = os.path.join('models', 'bert_phishing')


class BertPhishingDetector:
    def __init__(self, model_path_or_name=None, device=None):
        """
        Initialize BERT Phishing Detector.
        Checks for local fine-tuned model first, then falls back to pre-trained transformer.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if model_path_or_name is not None:
            self.model_name = model_path_or_name
        elif os.path.exists(LOCAL_BERT_PATH) and os.path.exists(os.path.join(LOCAL_BERT_PATH, 'config.json')):
            self.model_name = LOCAL_BERT_PATH
            print(f"[BERT] Loading local fine-tuned model from '{LOCAL_BERT_PATH}' on {self.device}...")
        else:
            self.model_name = DEFAULT_PRETRAINED_MODEL
            print(f"[BERT] Loading pre-trained Transformer '{self.model_name}' on {self.device}...")

        # Load Tokenizer & Model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[BERT] Successfully loaded model on {self.device}!")
        except Exception as e:
            print(f"[BERT Warning] Could not load '{self.model_name}': {e}")
            raise e

    def _extract_cues(self, text):
        """Extract explainable phishing cues and trigger patterns from raw email text."""
        cues = []
        lower = text.lower()

        urgency_patterns = ['immediately', 'within 24 hours', 'urgent', 'act now', 'suspended', 'terminate', 'final notice', 'deadline', 'lock your account']
        credential_patterns = ['verify your password', 'confirm your identity', 'security update', 'enter your credentials', 'reset your password', 'login credentials', 'billing update']
        financial_patterns = ['wire transfer', 'crypto', 'bitcoin', 'invoice attached', 'refund payment', 'unclaimed funds', 'lottery winner', 'million dollar']
        url_patterns = re.findall(r'https?://\S+|www\.\S+', text)
        ip_patterns = re.findall(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text)

        for pat in urgency_patterns:
            if pat in lower:
                cues.append(f"Urgency / Threat phrase: '{pat}'")
                break

        for pat in credential_patterns:
            if pat in lower:
                cues.append(f"Credential harvesting indicator: '{pat}'")
                break

        for pat in financial_patterns:
            if pat in lower:
                cues.append(f"Financial / Prize bait indicator: '{pat}'")
                break

        if ip_patterns:
            cues.append(f"Direct IP URL detected: '{ip_patterns[0]}'")
        elif url_patterns:
            cues.append(f"Embedded link detected: '{url_patterns[0][:40]}...'")

        if len(re.findall(r'!', text)) >= 3:
            cues.append("Excessive exclamation marks (psychological pressure)")

        return cues

    def predict(self, text, max_length=512):
        """
        Analyze a single email text with BERT.
        Returns a dictionary containing prediction, probabilities, threat level, and cues.
        """
        if not text or not str(text).strip():
            return {
                "prediction": "LEGITIMATE",
                "is_phishing": False,
                "confidence": 1.0,
                "phishing_prob": 0.0,
                "legit_prob": 1.0,
                "risk_level": "SAFE",
                "cues": ["Empty content"],
                "model_name": self.model_name
            }

        text_str = str(text)
        inputs = self.tokenizer(
            text_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

        # Handle class indexing based on model architecture
        num_classes = len(probs)
        if num_classes == 2:
            # Binary classification: 0 = Legit, 1 = Phishing
            legit_prob = float(probs[0])
            phishing_prob = float(probs[1])
        elif num_classes == 4:
            # Cybersectony multilabel: 0 = legit email, 1 = phishing url/email, 2 = legit url, 3 = phishing alt
            legit_prob = float(probs[0] + probs[2])
            phishing_prob = float(probs[1] + probs[3])
            # Normalize to sum to 1.0
            total = legit_prob + phishing_prob + 1e-12
            legit_prob /= total
            phishing_prob /= total
        else:
            phishing_prob = float(probs[-1])
            legit_prob = 1.0 - phishing_prob

        is_phishing = phishing_prob >= 0.50
        prediction = "PHISHING" if is_phishing else "LEGITIMATE"
        confidence = phishing_prob if is_phishing else legit_prob

        # Assess risk level
        if phishing_prob >= 0.85:
            risk_level = "CRITICAL PHISHING RISK"
        elif phishing_prob >= 0.50:
            risk_level = "SUSPICIOUS / POTENTIAL PHISHING"
        elif phishing_prob >= 0.20:
            risk_level = "LOW RISK (CHECK SENDER)"
        else:
            risk_level = "SAFE (LEGITIMATE)"

        cues = self._extract_cues(text_str)

        return {
            "prediction": prediction,
            "is_phishing": is_phishing,
            "confidence": confidence,
            "phishing_prob": phishing_prob,
            "legit_prob": legit_prob,
            "risk_level": risk_level,
            "cues": cues,
            "model_name": self.model_name
        }

    def predict_batch(self, texts, batch_size=32, max_length=256):
        """Batch inference for evaluating test datasets efficiently."""
        all_preds = []
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            for p in probs:
                if len(p) == 2:
                    p_phish = float(p[1])
                elif len(p) == 4:
                    p_legit = float(p[0] + p[2])
                    p_phish = float(p[1] + p[3])
                    tot = p_legit + p_phish + 1e-12
                    p_phish /= tot
                else:
                    p_phish = float(p[-1])

                all_probs.append(p_phish)
                all_preds.append(1 if p_phish >= 0.50 else 0)

        return np.array(all_preds), np.array(all_probs)


# Global singleton detector for fast reuse in CLI and Streamlit
_detector_instance = None

def get_bert_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BertPhishingDetector()
    return _detector_instance


if __name__ == '__main__':
    print("\n--- Testing BERT Phishing Detector ---")
    detector = get_bert_detector()

    test_samples = [
        "URGENT: Your bank account will be suspended within 24 hours. Click here to verify your password immediately: http://192.168.1.1/login",
        "Hi John, hope you are doing well. Please find attached the notes from our team sync yesterday. Let me know if anything needs revision.",
        "Dear customer, an unauthorized login attempt was detected from Moscow. Please verify your credentials immediately to avoid service interruption."
    ]

    for i, s in enumerate(test_samples, 1):
        res = detector.predict(s)
        print(f"\n[Sample {i}] \"{s[:70]}...\"")
        print(f"  Result      : {res['prediction']} ({res['risk_level']})")
        print(f"  Confidence  : {res['confidence']:.2%}")
        print(f"  Phishing %  : {res['phishing_prob']:.2%}")
        print(f"  Cues        : {res['cues']}")
