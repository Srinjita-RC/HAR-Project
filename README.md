# Human Activity Recognition (HAR) 

An end-to-end pipeline that recognizes human physical activities from smartphone sensor data (UCI HAR Dataset). It covers preprocessing, dimensionality reduction, supervised classification, unsupervised clustering, and evaluation.

## Project Overview

This project classifies six physical activities from accelerometer and gyroscope readings collected by a waist-mounted smartphone. It compares a simple baseline, classical machine learning models and neural networks, and also explores unsupervised activity grouping with standard and hybrid clustering methods.

**Activities recognized:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying

**Key contributions:**

- A modular pipeline: data loading → preprocessing → feature reduction → classification & clustering → evaluation
- A custom PyTorch MLP classifier (two hidden layers) trained on PCA-reduced features, reaching **92.91%** test accuracy
- A **Hybrid Distance-Filtered KMeans** clustering experiment that removes outliers before fitting centroids

## Project Structure

```
HAR-Project/
│
├── HAR_Code.ipynb     # Full pipeline: preprocessing, classification, neural network, clustering
├── UCI HAR Dataset/   # Dataset (downloaded separately, see below)
└── README.md
```

## Dataset

[UCI Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

| Attribute | Value |
|---|---|
| Subjects | 30 volunteers (age 19–48) |
| Device | Samsung Galaxy S II, waist-mounted |
| Sensors | 3-axis accelerometer + 3-axis gyroscope |
| Sampling rate | 50 Hz |
| Windowing | 2.56 s fixed-width windows (128 samples/window) |
| Labeling | Manual, via synchronized video annotation |
| Records | 10,299 total (7,352 train / 2,947 test) |
| Features | 561 original → 215 after cleaning → 106 after PCA |
| Missing values | None |

## Preprocessing Pipeline

1. **Data Cleaning**: Removed 84 duplicate features and 262 highly correlated features (correlation > 95%), reducing 561 features to 215. The same features are removed from both train and test sets.
2. **Normalization**: Standard scaling (zero mean, unit variance).
3. **Dimensionality Reduction**: PCA keeping ≥ 95% of the total variance, reducing 215 → **106** components.
4. **Activity Labeling**: Integer labels are mapped to activity names for readability.

The same preprocessed, PCA-reduced feature matrix is used for every classifier and clustering model, so performance differences reflect the algorithms and not the data handling.

## Classification

### Models

- **Rule-Based Classifier**: assigns each sample to the activity whose mean feature value (computed on training data) is closest. Serves as a simple baseline.
- **Decision Tree**: interpretable, but prone to overfitting on correlated signals.
- **Gaussian Naive Bayes**: assumes independent, normally distributed features.
- **SVM (RBF kernel)**: learns non-linear decision boundaries.
- **MLP (scikit-learn)**: multi-layer perceptron (128, 64) with ReLU activations.
- **Custom PyTorch Neural Network**: architecture below.

### Results (UCI HAR test set, 2,947 samples)

| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| Rule-Based Classifier | 28.47 | 0.271 | 0.284 | 0.269 |
| Decision Tree | 71.06 | 0.711 | 0.705 | 0.706 |
| Naive Bayes | 80.42 | 0.803 | 0.797 | 0.797 |
| MLP (scikit-learn) | 92.70 | 0.925 | 0.925 | 0.925 |
| Custom PyTorch Neural Network | 92.91 | 0.927 | 0.927 | 0.927 |
| **SVM (RBF Kernel)** | **94.30** | **0.942** | **0.941** | **0.941** |

**Takeaways:**

- The SVM and the neural networks clearly outperform the rule-based, Decision Tree and Naive Bayes models, showing that non-linear models suit inertial sensor data.
- The custom PyTorch network (92.91%) comes close to the SVM (94.30%).
- Laying is recognized almost perfectly (F1 ≈ 1.0). Most errors happen between similar activities: sitting vs. standing, and the three walking types.
- The rule-based model (28.47%) is only slightly above random guessing (~17% for 6 classes), which confirms the task needs non-linear models.

### Custom PyTorch Neural Network Architecture

- Input layer: 106 PCA features
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU
- Output layer: 6 classes
- Optimizer: Adam (lr = 0.001) | Loss: CrossEntropyLoss | Epochs: 30 | Batch size: 32

**Note:** Training loss falls to almost 0 by epoch ~25 while test accuracy stays near 93%, which suggests mild overfitting. Dropout, early stopping and a validation split are natural next steps.

## Clustering

All clustering methods run on the same PCA-reduced feature space (k = 6 for K-Means, Agglomerative and the hybrid method).

- **K-Means**: centroid-based partitioning.
- **Agglomerative Clustering**: hierarchical clustering with Ward linkage.
- **DBSCAN**: density-based clustering (eps = 5, min_samples = 10) that flags low-density points as outliers.
- **Hybrid Distance-Filtered KMeans (proposed)**:
  1. **Outlier filtering**: compute each sample's average distance to all others and drop the top 10% (90th-percentile threshold).
  2. **KMeans**: fit K-Means on the filtered core data.
  3. **Label assignment**: assign every point, including the filtered outliers, to its nearest centroid.

### Results

| Method | WCSS ↓ | Silhouette ↑ | Davies-Bouldin ↓ | Dunn Index ↑ |
|---|---|---|---|---|
| K-Means | 972,200 | 0.081 | 3.041 | 0.063 |
| Agglomerative (Ward) | 1,003,917 | 0.052 | 3.527 | 0.108 |
| DBSCAN | 1,502,851 | N/A | N/A | N/A |
| Hybrid Distance-Filtered KMeans | 981,730 | 0.074 | 3.170 | 0.078 |

All methods produce heavily overlapping clusters (very low Silhouette, high Davies-Bouldin). This is expected because several activities (e.g. sitting and standing) give very similar sensor signals. The hybrid method gave results close to standard K-Means and did **not** clearly improve on it, so it is best seen as an exploratory idea. DBSCAN did not produce a valid multi-cluster result on this feature space, so its cluster-quality metrics are not available (N/A).

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score (macro-averaged), and confusion matrices
- **Clustering**: Within-Cluster Sum of Squares (WCSS), Silhouette score, Davies-Bouldin index, Dunn index

## Installation & Usage

The code was developed and run in **Google Colab** (Python 3).

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) as a ZIP file.
2. Open `HAR_Code.ipynb` in Google Colab (or Jupyter).
3. Run the cells in order. When prompted, upload the `UCI HAR Dataset.zip` file. The notebook extracts and loads it automatically.

**Dependencies:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, torch

## Implementation Details

- **Language & libraries**: Python, NumPy, Pandas, scikit-learn (preprocessing, PCA, baseline classifiers, clustering), PyTorch (custom neural network), Matplotlib & Seaborn (visualization)
- **Environment**: Google Colab notebook
- **Deployment**: models are evaluated offline only. The PyTorch model could be exported (e.g. to ONNX) for real-time use on Android or other edge devices.

## Limitations

- Results use the single train/test split provided with the dataset, and subjects are not cross-validated.
- The neural network shows signs of mild overfitting (see note above).
- Clustering scores are low because similar activities overlap in feature space.

## Future Work

- CNN-LSTM and transformer models to capture temporal sequences and reduce errors at activity transitions
- Regularization (dropout, early stopping) and a validation split for the neural network
- Personalization and domain adaptation across users, devices and sensor placements
- Multimodal HAR using extra sensors such as magnetometers
- Explainable AI (feature importance, attention maps) and model compression for small wearable devices

## Reference

UCI Machine Learning Repository: Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The full report, *Human Activity Recognition using Data Mining, Machine Learning, and Deep Learning*, contains the complete literature review and reference list.

## Authors

- **Khyati Prashant Jetly**: Dept. of Computer Science, BITS Pilani Dubai
- **Srinjita Roy Chowdhury**: Dept. of Computer Science, BITS Pilani Dubai
- **Soorya Kiran Kakkarayil**: Dept. of Computer Science, BITS Pilani Dubai

## Institution

Birla Institute of Technology and Science, Pilani, Dubai Campus
