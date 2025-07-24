# Fake News Detection: Traditional ML vs Transformer Models

This project focuses on detecting fake news using state-of-the-art Transformer-based models, highlighting their superiority over traditional machine learning classifiers.

## Project Overview

- **Goal**: Classify news articles as real or fake using NLP techniques.
- **Dataset**: [Kaggle Fake News Dataset by Emine Bozkus](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **Models Used**:
  - Transformer Models: BERT, ELECTRA, RoBERTa
  - Traditional Models: Logistic Regression, Naive Bayes, SGD, KNN
- **Best Result**: BERT achieved 99.50% accuracy, F1-score 0.9947, ROC-AUC 0.9997.

## Key Features

- Text preprocessing: cleaning, lowercasing, stopword removal, regex for HTML and links
- Label encoding: Fake = 0, Real = 1
- TF-IDF vectorization for ML models
- Hugging Face Transformers for fine-tuned deep learning models
- Evaluation metrics: Accuracy, F1-Score, ROC-AUC

## Performance Comparison

The results clearly demonstrate that Transformer-based models significantly outperform traditional machine learning classifiers in fake news detection.

### Transformer Models (Top Performance)

| Model      | Accuracy | ROC-AUC | F1 Score |
|------------|----------|---------|----------|
| **BERT**   | **99.50%**   | **0.9997**  | **0.9947**   |
| RoBERTa    | 99.25%   | 0.9962  | 0.9920   |
| ELECTRA    | 98.75%   | 0.9991  | 0.9868   |

> These models leverage contextual embeddings and deep attention mechanisms, making them ideal for handling nuanced language in fake news detection.

### Traditional ML Models (Baseline)

| Model              | Accuracy | ROC-AUC | F1 Score |
|--------------------|----------|---------|----------|
| K-Nearest Neighbors| 87.75%   | 0.9451  | 0.8832   |
| Naive Bayes        | 92.50%   | 0.9771  | 0.9268   |
| Logistic Regression| 95.50%   | 0.9913  | 0.9519   |
| SGD Classifier     | 97.00%   | 0.9950  | 0.9683   |

> Traditional models serve as strong baselines, but fall short in capturing semantic complexity compared to BERT-family models.

## Tech Stack

- Python
- Hugging Face Transformers
- Scikit-learn
- Pandas / Numpy / NLTK
- Google Colab (GPU)

## File Structure

```
fake-news-detection/
├── fake-news-detection.ipynb     # Main code
├── README.md                     # Project documentation
```

## References

- Vaswani et al., 2017 – Attention is All You Need  
- Devlin et al., 2019 – BERT  
- Clark et al., 2020 – ELECTRA  
- Liu et al., 2019 – RoBERTa  
- Bozkus, E. (2022) – Kaggle Dataset