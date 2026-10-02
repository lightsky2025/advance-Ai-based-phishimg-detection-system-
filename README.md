# 🛡️ AI-Based Phishing Detection System (BERT & ML)

An intelligent, multi-layered cybersecurity system designed to detect and prevent phishing attacks across **Emails** and **URLs**. Powered by state-of-the-art **BERT Transformers** and comparative **Machine Learning Baselines**.

---

## 🚀 Key Improvements & Highlights: BERT Integration

### Why BERT Achieves Better Accuracy:
1. **Bidirectional Contextual Attention:** Unlike TF-IDF (which treats words as independent bags), BERT's self-attention mechanism processes words in relation to all other words in a sentence bidirectionally.
2. **Resilience to Obfuscation:** Attackers often replace trigger words (e.g. swapping *"verify password"* with *"confirm credentials"*). Classic TF-IDF models often fail on these substitutions, while BERT understands semantic intent and urgency patterns.
3. **Subword WordPiece Tokenization:** Captures typosquatting, leetspeak, and character substitutions used by phishing attackers.

### 📊 Benchmark Performance Comparison (Test Dataset)

| Model | Architecture | Accuracy | Precision | Recall (Phishing) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **BERT Transformer** | **Deep Self-Attention (SOTA)** | **98.60%** | **96.24%** | **100.00%** | **98.08%** | **0.9991** |
| Random Forest | TF-IDF (5007 features) | 96.43% | 94.04% | 97.06% | 95.52% | 0.9937 |
| Logistic Regression | TF-IDF + Linear | 95.68% | 93.63% | 95.49% | 94.55% | 0.9911 |
| Naive Bayes | TF-IDF + Bayes | 80.76% | 70.75% | 86.87% | 77.99% | 0.9084 |
| URL Random Forest | Lexical & Structural Features | 81.47% | 84.04% | 77.70% | 80.74% | 0.9053 |

> **Key Takeaway:** BERT achieved **100% recall on phishing emails** in testing, effectively eliminating false negatives that slipped past traditional models.

---

## 📁 Project Architecture

```
├── bert_classifier.py        # Core BERT inference engine, threat scoring & cue extraction
├── train_bert.py             # BERT fine-tuning script with PyTorch & HuggingFace
├── app.py                    # Streamlit web dashboard with BERT & ML engines
├── predict.py                # Interactive CLI tool for testing emails and URLs
├── evaluation.py             # Side-by-side benchmark comparison & chart generator
├── preprossing.py            # Data cleaning and label encoding pipeline
├── feature.py                # Feature engineering (TF-IDF + lexical metadata)
├── training.py               # Baseline ML training (RF, LR, NB)
├── first.py                  # Initial exploratory data analysis & visualization
```

---

## 💻 How to Run

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive UI:
- **Email Detection Tab:** Analyze emails with BERT or Random Forest. Includes pre-loaded samples and threat cue extraction.
- **Side-by-Side Comparison Tab:** Run BERT and Random Forest concurrently on the same email.
- **URL Detection Tab:** Inspect URLs with structural feature analysis.
- **Benchmarks Tab:** View live accuracy tables and comparison charts.

### 2. Run the Interactive Command-Line Tool
```bash
python predict.py
```

### 3. Run Model Benchmarks & Generate Plots
```bash
python evaluation.py
```

### 4. Fine-Tune BERT on Your Own Dataset
```bash
python train_bert.py --epochs 2 --batch_size 16 --sample_size 2000
```
