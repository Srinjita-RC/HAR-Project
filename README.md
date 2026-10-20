# Human Activity Recognition (HAR) using Data Mining, Machine Learning & Deep Learning

A complete, modular, end-to-end pipeline for recognizing human physical activities from smartphone sensor data using the UCI HAR Dataset. The pipeline covers preprocessing, dimensionality reduction, supervised classification, unsupervised clustering, and evaluation.

## 📖 Project Overview

This project classifies six physical activities using accelerometer and gyroscope readings collected from a waist-mounted smartphone. It compares interpretable baselines, classical machine learning models, and deep neural networks, and also explores unsupervised activity grouping with standard and hybrid clustering methods.

**Activities recognized:**

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

**Key contributions:**

- A modular pipeline: data acquisition → preprocessing → feature reduction → classification & clustering → evaluation & visualization
- A custom PyTorch MLP classifier (two hidden layers) trained on PCA-reduced features
- A **Hybrid Distance-Filtered KMeans** clustering method that removes outliers before fitting centroids

## 🗂️ Project Structure

```
HAR-Project/
│
├── main.py               # Main pipeline runner
├── preprocessing.py      # Data cleaning, normalization, PCA
├── classification.py     # ML classification models
├── clustering.py         # Clustering algorithms
├── neural_network.py     # PyTorch deep learning model
└── requirements.txt      # Dependencies
```

## 📊 Dataset

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
| Features | 561 original → 106 after cleaning + PCA |
| Missing values | None |

## ⚙️ Preprocessing Pipeline

1. **Data Cleaning**: Removed duplicate samples, constant features, 84 duplicate features, and 262 highly correlated features (correlation threshold > 95%), leaving 215 features.
2. **Handling Missing Values**: Any missing values are imputed with the feature median (the UCI HAR dataset itself has none).
3. **Normalization**: Standard scaling (zero mean, unit variance).
4. **Dimensionality Reduction**: PCA retaining ≥ 95% of the total variance, reducing 215 → **106** components.
5. **Activity Labeling**: Integer labels are mapped to descriptive class names for interpretability.

The same preprocessed, PCA-reduced feature matrix is used for every classifier and clustering model, so performance differences reflect the algorithms rather than data handling.

## 🤖 Classification

### Models

- **Rule-Based Classifier**: assigns each sample to the activity whose mean total feature intensity (computed on training data) is closest. Serves as a lightweight baseline.
- **Decision Tree**: interpretable but prone to overfitting on correlated signals.
- **Gaussian Naive Bayes**: assumes independent, normally distributed features.
- **SVM (RBF kernel)**: learns non-linear decision boundaries.
- **MLP (scikit-learn)**: multi-layer perceptron with ReLU activations.
- **Proposed PyTorch Neural Network**: custom MLP (architecture below).

### Results (UCI HAR test set)

| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| Rule-Based Classifier | 28.50 | 0.27 | 0.28 | 0.26 |
| Decision Tree | 71.06 | 0.71 | 0.70 | 0.70 |
| Naive Bayes | 80.42 | 0.80 | 0.79 | 0.79 |
| MLP (scikit-learn) | 92.10 | 0.91 | 0.91 | 0.91 |
| Proposed PyTorch Neural Network | 92.70 | 0.92 | 0.92 | 0.92 |
| **SVM (RBF Kernel)** | **94.30** | **0.94** | **0.94** | **0.94** |

**Takeaways:**

- The SVM (RBF) and neural networks clearly outperform the rule-based, Decision Tree, and Naive Bayes baselines, confirming that non-linear models suit inertial sensor data.
- The proposed PyTorch network (92.7%) comes close to the SVM (94.3%).
- Remaining errors occur mostly between semantically similar activities (sitting vs. standing, and the three walking types), especially at transition windows.

### Proposed PyTorch Neural Network Architecture

- Input layer (106 PCA features)
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU
- Output layer: 6 classes
- Optimizer: Adam | Loss: CrossEntropyLoss | Epochs: 30 | Batch size: 32

## 🔍 Clustering

All clustering methods run on the same PCA-reduced feature space (k = 6 for K-Means and the hybrid method).

- **K-Means**: centroid-based partitioning.
- **Agglomerative Clustering**: hierarchical clustering with Ward linkage.
- **DBSCAN**: density-based clustering that flags low-density points as outliers.
- **Hybrid Distance-Filtered KMeans (proposed)**:
  1. **Outlier filtering**: compute each sample's average pairwise distance to all others and drop the samples above a threshold (e.g. the largest 10%).
  2. **KMeans integration**: fit KMeans on the filtered core data.
  3. **Label assignment**: assign every point, including filtered outliers, to its nearest centroid.

### Results

| Method | Silhouette | Davies-Bouldin ↓ | Dunn Index ↑ |
|---|---|---|---|
| K-Means | 0.08 | 3.04 | 0.06 |
| Agglomerative (Ward) | 0.05 | 3.52 | 0.10 |
| DBSCAN | 0 | 0 | 0 |
| Hybrid Distance-Filtered KMeans (proposed) | 0.07 | 3.17 | 0.07 |

All methods show fairly overlapping clusters (low Silhouette, high Davies-Bouldin), which is typical for noisy, high-dimensional wearable-sensor data. DBSCAN did not produce a valid multi-cluster result on this feature space, so its metrics are reported as 0.

## 📏 Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score (macro-averaged), AUC, and confusion matrices
- **Clustering**: Within-Cluster Sum of Squares (WCSS), Silhouette score, Davies-Bouldin index, Dunn index

## 🛠️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/khyati2107/HAR-Project.git
cd HAR-Project
```

### 2. Create a virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the UCI HAR Dataset from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract it so your folder looks like:

```
HAR-Project/
└── UCI HAR Dataset/
    ├── train/
    └── test/
```

### 5. Run the pipeline

```bash
python main.py --data_path "UCI HAR Dataset"
```

## 💻 Implementation Details

- **Language & libraries**: Python 3.8+, NumPy, Pandas, scikit-learn (preprocessing, PCA, baseline classifiers, clustering), PyTorch (proposed neural network), Matplotlib & Seaborn (visualization)
- **Environment**: Windows 11 and Ubuntu 22.04, run in Jupyter Notebook and Visual Studio Code
- **Hardware used**: Intel Core i5 processor, 16 GB RAM, ≥ 10 GB free storage. No special server required; GPU acceleration (e.g. NVIDIA RTX 30-series) is optional for larger networks
- **Deployment**: models are currently evaluated offline. The modular design allows exporting the PyTorch model (e.g. to ONNX) for real-time use on Android or other edge devices

## 📦 Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
scipy
```

## 📈 Output Files

All figures are saved automatically to the `outputs/` folder:

- `pca_explained_variance.png`: cumulative explained variance (106 components capture > 95%)
- `dataset_overview.png`: train vs. test feature distribution and activity distributions
- `clustering_results.png`
- `confusion_matrix_*.png`: one per model
- `model_accuracy_comparison.png`
- `neural_network_training_loss.png`
- `neural_network_confusion_matrix.png`

## 🔮 Future Work

- CNN-LSTM and transformer models to capture temporal sequences and reduce transition-window errors
- Personalization and domain adaptation across users, devices, and sensor placements
- Multimodal HAR by adding sensors such as magnetometers or environmental data
- Explainable AI (attention maps, feature importance) and model compression for deployment on small wearable and mobile devices

## 📚 Reference

UCI Machine Learning Repository: Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The full report, *Human Activity Recognition using Data Mining, Machine Learning, and Deep Learning*, contains the complete literature review and reference list.

## 👥 Authors

- **Khyati Prashant Jetly**: Dept. of Computer Science, BITS Pilani Dubai
- **Srinjita Roy Chowdhury**: Dept. of Computer Science, BITS Pilani Dubai
- **Soorya Kiran Kakkarayil**: Dept. of Computer Science, BITS Pilani Dubai

## 🏫 Institution

Birla Institute of Technology and Science, Pilani, Dubai Campus
