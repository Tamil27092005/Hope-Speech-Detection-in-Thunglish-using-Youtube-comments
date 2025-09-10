# Hope-Speech-Detection-in-Thunglish-using-Youtube-comments
This repository implements Hope Speech Detection on Tamil-English code-mixed text (Thunglish) using XLM-RoBERTa, enhanced with hybrid preprocessing (dictionary-based normalization + Levenshtein fuzzy matching).  The pipeline is designed to handle social media text variations, spelling mistakes

**Features**

Hybrid Preprocessing

Exact dictionary normalization

Levenshtein distance–based fuzzy matching

Handles stretched words & Tamil-English mix

**Model**

XLM-RoBERTa (base, ~270M parameters)

Supports 100+ languages (Tamil + English included)

Optimized training with early stopping

**Dataset
**
~14k training samples, ~1.7k validation, ~1.7k test

Labels: Hope_speech, Non_hope_speech

**Results**

Strong multilingual performance on Thunglish Hope Speech Detection

Handles noisy social media text effectively

{
  "original_text": "Semma video! This gives me hope and courage to achieve my dreams",
  "processed_text": "awesome video! This gives me hope and courage to achieve my dreams",
  "predicted_class": "Hope_speech",
  "confidence": 0.87
}
**
Results Summary**

Test Accuracy: ~67%

Normalization Coverage: ~3.6% words normalized

Model: XLM-RoBERTa Base (~270M parameters)

Languages Supported: 100+ (Tamil + English included)

**Advantages over DistilBERT:**

Better multilingual understanding

Robust to spelling variations

Stronger handling of Tamil-English code-mixing

Contextual embeddings for noisy social media text
