"""
ML Models Module for Phishing Detection System
Implements XGBoost, Random Forest, and BERT-based models
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

from config import MODEL_CONFIG, TRAINING_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')


class EmailClassifier:
    """
    Email phishing classifier using XGBoost and Random Forest
    """
    
    def __init__(self, use_xgboost: bool = True):
        """
        Initialize the email classifier
        
        Args:
            use_xgboost: If True, use XGBoost; otherwise use Random Forest
        """
        self.use_xgboost = use_xgboost
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model based on preference
        if use_xgboost:
            params = TRAINING_CONFIG["xgboost_params"]
            self.model = xgb.XGBClassifier(**params)
        else:
            params = TRAINING_CONFIG["random_forest_params"]
            self.model = RandomForestClassifier(**params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'EmailClassifier':
        """
        Train the classifier
        
        Args:
            X: Feature matrix
            y: Target labels (0 = legitimate, 1 = phishing)
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates for each class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
    
    def save(self, path: str):
        """Save the model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'use_xgboost': self.use_xgboost,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str) -> 'EmailClassifier':
        """Load the model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.use_xgboost = model_data['use_xgboost']
        self.is_fitted = model_data['is_fitted']
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'type': 'XGBoost' if self.use_xgboost else 'Random Forest',
            'is_fitted': self.is_fitted,
            'n_features': self.model.n_features_in_ if self.is_fitted else None
        }
        
        if self.is_fitted:
            if self.use_xgboost:
                info['n_estimators'] = self.model.n_estimators
                info['max_depth'] = self.model.max_depth
                info['learning_rate'] = self.model.learning_rate
            else:
                info['n_estimators'] = self.model.n_estimators
                info['max_depth'] = self.model.max_depth
        
        return info


class BERTURLClassifier:
    """
    BERT-based URL and text classifier for phishing detection
    Uses transfer learning with pre-trained BERT model
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Name of the pre-trained BERT model
        """
        self.model_name = model_name or MODEL_CONFIG["bert_model_name"]
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_fitted = False
        
        # Try to import transformers
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            self.BertTokenizer = BertTokenizer
            self.BertForSequenceClassification = BertForSequenceClassification
            self.transformers_available = True
        except ImportError:
            self.transformers_available = False
            print("Warning: transformers library not available. BERT model will not be functional.")
    
    def _init_model(self, num_labels: int = 2):
        """Initialize the BERT model and tokenizer"""
        if not self.transformers_available:
            return
        
        self.tokenizer = self.BertTokenizer.from_pretrained(self.model_name)
        self.model = self.BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = None, batch_size: int = None, 
              learning_rate: float = None) -> 'BERTURLClassifier':
        """
        Fine-tune BERT model on URL/text data
        
        Args:
            texts: List of text samples
            labels: List of labels (0 = legitimate, 1 = phishing)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Self for method chaining
        """
        if not self.transformers_available:
            print("BERT training skipped - transformers not available")
            return self
        
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        # Set defaults from config
        config = TRAINING_CONFIG["bert_config"]
        epochs = epochs or config["epochs"]
        batch_size = batch_size or config["batch_size"]
        learning_rate = learning_rate or config["learning_rate"]
        max_length = config["max_length"]
        
        # Initialize model
        self._init_model(num_labels=2)
        
        # Create dataset
        class URLDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        dataset = URLDataset(texts, labels, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        from transformers import get_linear_schedule_with_warmup

        try:
            from torch.optim import AdamW
        except ImportError:
            from transformers import AdamW
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels for texts/URLs
        
        Args:
            texts: List of text/URL samples
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted or not self.transformers_available:
            # Return random predictions if model not available
            return np.random.randint(0, 2, size=len(texts))
        
        import torch
        
        self.model.eval()
        predictions = []
        
        config = TRAINING_CONFIG["bert_config"]
        max_length = config["max_length"]
        batch_size = config["batch_size"]
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for texts/URLs
        
        Args:
            texts: List of text/URL samples
            
        Returns:
            Probability estimates for each class
        """
        if not self.is_fitted or not self.transformers_available:
            # Return uniform probabilities if model not available
            return np.ones((len(texts), 2)) * 0.5
        
        import torch
        
        self.model.eval()
        probabilities = []
        
        config = TRAINING_CONFIG["bert_config"]
        max_length = config["max_length"]
        batch_size = config["batch_size"]
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save(self, path: str):
        """Save the model"""
        if self.is_fitted and self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
    
    def load(self, path: str) -> 'BERTURLClassifier':
        """Load the model"""
        if not self.transformers_available:
            return self
        
        self._init_model()
        self.model = self.BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = self.BertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        self.is_fitted = True
        
        return self


class EnsemblePhishingDetector:
    """
    Ensemble model combining XGBoost, Random Forest, and BERT
    for robust phishing detection
    """
    
    def __init__(self):
        """Initialize ensemble detector"""
        self.xgboost_classifier = EmailClassifier(use_xgboost=True)
        self.random_forest_classifier = EmailClassifier(use_xgboost=False)
        self.bert_classifier = BERTURLClassifier()
        self.is_fitted = False
        
        # Weights for ensemble (can be optimized)
        self.weights = {
            'xgboost': 0.4,
            'random_forest': 0.3,
            'bert': 0.3
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              texts: List[str] = None) -> 'EnsemblePhishingDetector':
        """
        Train all models in the ensemble
        
        Args:
            X: Feature matrix for ML models
            y: Target labels
            texts: Optional text data for BERT model
            
        Returns:
            Self for method chaining
        """
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgboost_classifier.train(X, y)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.random_forest_classifier.train(X, y)
        
        # Train BERT if texts provided
        if texts is not None and self.bert_classifier.transformers_available:
            print("Training BERT model...")
            self.bert_classifier.train(texts, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, texts: List[str] = None) -> np.ndarray:
        """
        Predict using weighted ensemble
        
        Args:
            X: Feature matrix
            texts: Optional text data for BERT
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Get predictions from each model
        xgb_proba = self.xgboost_classifier.predict_proba(X)[:, 1]
        rf_proba = self.random_forest_classifier.predict_proba(X)[:, 1]
        
        # Calculate weighted probability
        ensemble_proba = (
            self.weights['xgboost'] * xgb_proba + 
            self.weights['random_forest'] * rf_proba
        )
        
        # Add BERT if available
        if texts is not None and self.bert_classifier.is_fitted:
            bert_proba = self.bert_classifier.predict_proba(texts)[:, 1]
            ensemble_proba += self.weights['bert'] * bert_proba
        
        # Convert to binary predictions
        return (ensemble_proba >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray, texts: List[str] = None) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble
        
        Args:
            X: Feature matrix
            texts: Optional text data for BERT
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Get probabilities from each model
        xgb_proba = self.xgboost_classifier.predict_proba(X)[:, 1]
        rf_proba = self.random_forest_classifier.predict_proba(X)[:, 1]
        
        # Calculate weighted probability
        ensemble_proba = (
            self.weights['xgboost'] * xgb_proba + 
            self.weights['random_forest'] * rf_proba
        )
        
        # Add BERT if available
        if texts is not None and self.bert_classifier.is_fitted:
            bert_proba = self.bert_classifier.predict_proba(texts)[:, 1]
            ensemble_proba += self.weights['bert'] * bert_proba
        
        # Return as 2D array [prob_legitimate, prob_phishing]
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def get_individual_predictions(self, X: np.ndarray, 
                                    texts: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual models
        
        Args:
            X: Feature matrix
            texts: Optional text data for BERT
            
        Returns:
            Dictionary of model names to their predictions
        """
        predictions = {
            'xgboost': self.xgboost_classifier.predict(X),
            'random_forest': self.random_forest_classifier.predict(X)
        }
        
        if texts is not None and self.bert_classifier.is_fitted:
            predictions['bert'] = self.bert_classifier.predict(texts)
        
        return predictions
    
    def save(self, base_path: str):
        """Save all models"""
        # Create directory if it doesn't exist
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.xgboost_classifier.save(f"{base_path}_xgboost.pkl")
        self.random_forest_classifier.save(f"{base_path}_random_forest.pkl")
        if self.bert_classifier.is_fitted:
            self.bert_classifier.save(f"{base_path}_bert")
    
    def load(self, base_path: str) -> 'EnsemblePhishingDetector':
        """Load all models"""
        xgb_path = f"{base_path}_xgboost.pkl"
        rf_path = f"{base_path}_random_forest.pkl"
        
        if Path(xgb_path).exists():
            self.xgboost_classifier.load(xgb_path)
        if Path(rf_path).exists():
            self.random_forest_classifier.load(rf_path)
        
        bert_path = f"{base_path}_bert"
        if Path(bert_path).exists():
            self.bert_classifier.load(bert_path)
        
        self.is_fitted = True
        return self
    
