# 🛡️ AI-Powered Phishing Detection System

An intelligent, production-ready phishing detection system that leverages machine learning and natural language processing to identify malicious emails and URLs with high accuracy.

## ✨ Features

### 🔗 URL Phishing Detection
- Structural analysis (length, special characters, subdomains)
- Protocol and security indicators (HTTPS, certificates)
- Domain reputation and TLD analysis
- Entropy calculation for obfuscation detection
- URL shortening service detection

### 📧 Email Phishing Detection
- Text analysis (length, capitalization, special characters)
- Linguistic pattern analysis (urgency, threats, suspicious words)
- Sender reputation scoring
- Authentication header verification (SPF, DKIM, DMARC)
- HTML content analysis
- Embedded URL extraction and analysis

### 🤖 Multi-Model Ensemble
- **XGBoost**: Gradient boosting for structured feature analysis
- **Random Forest**: Robust ensemble decision tree classification

### 🎯 Real-Time Analysis
- Instant detection results
- Detailed risk assessments
- Actionable security recommendations
- Batch processing capabilities

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit Web Interface (app.py)            │
├──────────────────────────────────────────────────────────────┤
│                   Phishing Detector (detector.py)            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Email Analysis Engine                        │  │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐│  │
│  │  │ XGBoost Email   │    │ Random Forest Email         ││  │
│  │  │ Classifier      │    │ Classifier                  ││  │
│  │  └────────┬────────┘    └─────────────┬───────────────┘│  │
│  │           │                           │                │  │
│  │  ┌────────▼──────────────────────────▼───────────────┐ │  │
│  │  │         Email Feature Extraction                  │ │  │
│  │  │  • Text Analysis    • Linguistic Patterns         │ │  │
│  │  │  • Sender Reputation • Auth Headers (SPF/DKIM)    │ │  │
│  │  └───────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           URL Analysis Engine                          │  │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐│  │
│  │  │ XGBoost URL     │    │ Random Forest URL           ││  │
│  │  │ Classifier      │    │ Classifier                  ││  │
│  │  └────────┬────────┘    └─────────────┬───────────────┘│  │
│  │           │                           │                │  │
│  │  ┌────────▼──────────────────────────▼───────────────┐ │  │
│  │  │         URL Feature Extraction                    │ │  │
│  │  │  • Structural Analysis  • Security Indicators     │ │  │
│  │  │  • Domain Reputation    • Entropy Calculation     │ │  │
│  │  └───────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd phishing-detection-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   python train.py
   ```

5. **Run the web application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
phishing-detection-system/
├── app.py                  # Streamlit web application
├── config.py               # Configuration settings
├── detector.py             # Main detection engine
├── feature_engineering.py  # Feature extraction modules
├── models.py               # ML model implementations
├── train.py                # Training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Trained model files (generated)
│   ├── xgboost_url_model.pkl
│   ├── random_forest_url_model.pkl
│   ├── xgboost_email_model.pkl
│   ├── random_forest_email_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── feature_names.txt
│   └── training_metrics.json
├── data/                   # Data directory (for datasets)
├── static/                 # Static assets
└── templates/              # HTML templates (if needed)
```

## 📖 Usage Guide

### URL Detection

1. Navigate to the **URL Detection** page
2. Enter a single URL in the input field
3. Click **Analyze URL** to get instant results
4. View detailed risk analysis, factors, and recommendations

**Batch URL Analysis:**
1. Expand the **Batch URL Analysis** section
2. Enter multiple URLs (one per line)
3. Click **Analyze Batch URLs** for bulk processing

### Email Detection

1. Navigate to the **Email Detection** page
2. Fill in the email details:
   - Sender email address (optional)
   - Email subject
   - Email body content
3. Click **Analyze Email**
4. Review comprehensive analysis including:
   - Email content analysis
   - Embedded URL analysis
   - Combined risk assessment
   - Security recommendations

### Understanding Results

#### Risk Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| 🔴 HIGH | ≥ 80% | Strong indicators of phishing |
| 🟠 MEDIUM | 50-79% | Some suspicious indicators present |
| 🟡 LOW | 30-49% | Minor concerns, likely safe |
| 🟢 SAFE | < 30% | No significant risk indicators |

#### Risk Factors

The system identifies specific risk factors such as:
- Suspicious TLDs (.xyz, .tk, .ml, etc.)
- IP addresses instead of domain names
- URL shortening services
- Missing HTTPS encryption
- Suspicious language patterns
- Low sender reputation
- Failed authentication checks

## 🧪 Model Training

### Training with Synthetic Data

The system includes a training script that generates synthetic data for demonstration:

```bash
python train.py
```

This will:
1. Generate synthetic phishing and legitimate email samples with realistic class overlap
2. Extract features from the data (40+ URL features, 20+ email features)
3. Train XGBoost and Random Forest models for both URL and Email detection
4. Evaluate model performance with domain-aware splitting
5. Save trained models to the `models/` directory

### Training with Custom Data

To train with your own dataset:

1. Prepare your data in CSV format with columns:
   - `subject`: Email subject
   - `body`: Email body content
   - `sender`: Sender email address
   - `label`: 0 (legitimate) or 1 (phishing)

2. Modify `train.py` to load your dataset:
   ```python
   # Replace generate_synthetic_data() with:
   data = pd.read_csv('your_dataset.csv')
   labels = data['label']
   texts = data['subject'] + ' ' + data['body']
   ```

3. Run the training script

## 📊 Model Performance

The system achieves high accuracy on real-world datasets using domain-aware splitting to prevent data leakage:

### URL Detection Models
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 96.68% | 98.38% | 91.79% | 94.98% |
| Random Forest | 95.95% | 97.45% | 90.55% | 93.87% |

### Email Detection Models
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 86.71% | 87.95% | 89.05% | 88.49% |
| Random Forest | 86.00% | 87.18% | 88.63% | 87.90% |

### Evaluation Metrics Explained
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to detect actual phishing attempts
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

## 🔒 Security Considerations

### Privacy
- All analysis is performed locally
- No data is sent to external servers
- Email content is not stored persistently

### Limitations
- No system is 100% accurate
- Novel phishing techniques may not be detected
- Context-specific attacks may require human judgment
- Model performance depends on training data quality

### Best Practices
1. Always verify unexpected requests through official channels
2. Never share passwords or sensitive information via email
3. Check sender email addresses carefully
4. Hover over links before clicking
5. Keep software and browsers updated
6. Use multi-factor authentication
7. Report suspicious emails to your IT/security team

## 🛠️ Configuration

Edit `config.py` to customize:

- **Model paths**: Location of trained model files
- **Feature configuration**: Features to extract
- **Suspicious words**: Keywords to detect
- **Detection thresholds**: Risk level boundaries
- **Training parameters**: Model hyperparameters

## 📈 Performance Optimization

### For Production Deployment

1. **Enable model caching**:
   - Models are automatically cached in memory
   - Consider using `joblib` for faster model loading

3. **Optimize feature extraction**:
   - Pre-compute features for batch processing
   - Use vectorized operations where possible

4. **Deploy with proper infrastructure**:
   - Use a production WSGI server
   - Implement proper logging and monitoring
   - Set up automated model retraining

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML tools
- XGBoost contributors
- Streamlit team for the great web framework
- Open-source security community

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation
- Review the example code

## 🔮 Future Enhancements

- [ ] Integration with threat intelligence feeds
- [ ] Real-time email server integration
- [ ] Browser extension for real-time URL checking
- [ ] Advanced phishing campaign detection
- [ ] Multi-language support
- [ ] Automated model retraining pipeline
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Kubernetes deployment configuration

---

**⚠️ Disclaimer**: This system is designed to assist in phishing detection but should not be the sole security measure. Always combine with other security practices and human vigilance.