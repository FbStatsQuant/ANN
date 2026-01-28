# Artificial Neural Network Examples

Two complete implementations of ANNs for supervised learning tasks using TensorFlow/Keras.

## Contents

- **ANN_Classification.ipynb**: Binary classification for fraud detection
- **ANN_Reg.ipynb**: Regression for price prediction

## Classification Model

Binary classification on imbalanced fraud detection dataset.

### Features
- Handles severe class imbalance using SMOTE
- Custom ANN with LeakyReLU activation and Dropout regularization
- Comprehensive evaluation: ROC-AUC, confusion matrix, precision-recall curves

### Architecture
```
Input → Dense(64, LeakyReLU) → Dropout(0.3) →
Dense(32, LeakyReLU) → Dropout(0.3) →
Dense(16, LeakyReLU) → Dense(1, sigmoid)
```

### Key Techniques
- SMOTE oversampling for minority class
- StandardScaler for feature normalization
- Early stopping with validation monitoring
- Class weight balancing
- Binary cross-entropy loss with Adam optimizer

## Regression Model

Continuous value prediction comparing ANN vs Linear Regression baseline.

### Architecture
```
Input → Dense(128, LeakyReLU) → Dropout(0.3) →
Dense(64, LeakyReLU) → Dropout(0.2) →
Dense(32, LeakyReLU) → Dense(1, linear)
```

### Features
- Feature engineering with one-hot encoding
- MSE and MAE evaluation metrics
- Training/validation loss visualization
- Baseline comparison with linear regression

## Requirements

```
tensorflow>=2.13.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Installation

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

## Usage

### Classification
```python
# Load and run the notebook
jupyter notebook ANN_Classification.ipynb
```

Expected outputs: confusion matrix, classification report, ROC curve, precision-recall curve

### Regression
```python
# Load and run the notebook
jupyter notebook ANN_Reg.ipynb
```

Expected outputs: MSE/MAE metrics, loss curves, predictions vs actuals plot

## Model Training Details

### Classification
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout (0.3)
- **Early Stopping**: patience=5, monitor='val_loss'
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)

### Regression
- **Loss**: Mean Squared Error
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout (0.2-0.3)
- **Early Stopping**: patience=10, monitor='val_loss'
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)

## Data Requirements

### Classification
- Expects tabular data with mixed feature types
- Binary target variable
- Handles imbalanced classes automatically

### Regression
- Expects tabular data with categorical and numerical features
- Continuous target variable
- Includes preprocessing pipeline

## Notes

- Both models use LeakyReLU activation to avoid dying ReLU problem
- Dropout layers prevent overfitting
- StandardScaler applied to all features
- Models trained with early stopping to prevent overfitting

## License

MIT
