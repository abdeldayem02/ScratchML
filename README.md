# ScratchML

ScratchML is a Python library for implementing machine learning and deep learning models from scratch. The goal of this project is to provide a deeper understanding of how these algorithms work under the hood by building them without relying on high-level libraries like TensorFlow, PyTorch, or Scikit-learn.

## Features

### Supervised Learning Models (Classical ML)

#### Regression (Predicting Continuous Values)
- Linear Regression (With Gradient Descent & Normal Equation)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

#### Classification (Predicting Discrete Labels)
- Logistic Regression (Binary & Multi-class Classification)
- K-Nearest Neighbors (KNN) (Distance-based classification)
- Naïve Bayes (Gaussian & Multinomial)
- Decision Tree (Entropy, Gini Index)
- Random Forest (Ensemble Learning)
- Support Vector Machine (SVM) (Hard & Soft Margin, Kernel Trick)

#### Boosting & Advanced ML
- Gradient Boosting (AdaBoost, XGBoost, LightGBM)

### Unsupervised Learning Models (No Labels)

#### Clustering Algorithms
- K-Means Clustering (Centroid-based clustering)
- Hierarchical Clustering (Agglomerative & Divisive)
- DBSCAN (Density-Based Spatial Clustering)

#### Dimensionality Reduction & Feature Extraction
- Principal Component Analysis (PCA)
- Autoencoders (A neural network-based unsupervised model)

### Deep Learning Models

#### Basic Neural Networks
- Perceptron (The foundation of neural networks)
- Multi-Layer Perceptron (MLP) (Backpropagation from scratch)

#### Computer Vision
- Convolutional Neural Networks (CNNs) (With Conv, Pooling, Dropout)

#### Sequence Models
- Recurrent Neural Networks (RNNs) (Basic RNN for time-series/text)
- Long Short-Term Memory (LSTMs) (For NLP and sequential tasks)

#### Advanced Deep Learning
- Transformers (Self-Attention, Positional Encoding)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/ScratchML.git
cd ScratchML
```

Ensure you have Python 3.7+ installed along with the required dependencies:

```bash
pip install -r requirements.txt
```


## Project Structure

```
ScratchML/
├── supervised_learning/
│   ├── regression.py       # Contains regression models (Linear, Ridge, Lasso, Polynomial)
│   └── __init__.py         # Package initialization
├── deep_learning/          # (Planned) Deep learning models and utilities
├── tests/                  # Unit tests for the library
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Goals

1. **Educational Purpose**: This library is designed to help developers and students understand the inner workings of machine learning and deep learning algorithms.
2. **Extendability**: The library is modular, making it easy to add new models and features.
3. **Performance**: While the focus is on understanding, efforts are made to ensure the models are efficient and scalable.

## Contributing

Contributions are welcome! If you'd like to add new models, improve existing ones, or fix bugs, feel free to open a pull request or submit an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by the desire to learn and teach the fundamentals of machine learning and deep learning by building models from scratch.