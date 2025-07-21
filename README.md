# 🤖 Machine Learning & Deep Learning Projects

This repository features a diverse collection of projects ranging from classical ML techniques like SVM and PCA to deep learning-based gesture recognition. Each notebook showcases a unique application, demonstrating both theoretical understanding and practical implementation using Python and industry-standard libraries.

---

## 📁 Projects Included

### 1. 🛍️ Customer Segmentation using K-Means Clustering

**Goal:** Segment customers into behavioral clusters using K-Means for targeted marketing strategies.

- **Dataset:** Mall Customers Dataset ([Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python))
- **Techniques:** 
  - K-Means Clustering
  - Elbow Method
  - 2D Scatter Plot of clusters
- **Libraries:** pandas, seaborn, matplotlib, scikit-learn
- **Key Outcomes:**
  - Identified distinct customer segments (e.g., high-income low-spenders)
  - Used Elbow Method to choose optimal clusters (k=5)
  - Visual representation of clusters for business insights

---

### 2. ✋ Hand Gesture Recognition using CNN

**Goal:** Recognize static hand gestures using Convolutional Neural Networks (CNNs).

- **Dataset:** Hand Sign Digit Recognition Dataset (A-Z and digits)([Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog))
- **Techniques:**
  - Deep Learning using CNN (Sequential Model)
  - Data Augmentation with ImageDataGenerator
  - Real-time prediction on test gestures
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Key Outcomes:**
  - Achieved high accuracy on test set (>95%)
  - Trained CNN with data augmentation for generalization
  - Predicts gestures from images using live model inference

---

### 3. 🐶 Cats vs Dogs Classification using HOG & SVM

**Goal:** Classify images of cats and dogs using classical ML (HOG + SVM).([Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data))

- **Dataset:** Dogs vs Cats Dataset (Kaggle subset)
- **Techniques:**
  - Feature Extraction with HOG (Histogram of Oriented Gradients)
  - Classification using Support Vector Machines (SVM)
  - GridSearchCV for hyperparameter tuning
- **Libraries:** OpenCV, skimage, scikit-learn, matplotlib
- **Key Outcomes:**
  - Reached ~88–90% classification accuracy
  - Efficient image classification without deep learning
  - Demonstrated power of HOG descriptors in image recognition tasks

---

### 4. 🏠 House Price Prediction using PCA & Linear Regression

**Goal:** Predict housing prices using dimensionality reduction followed by linear regression.

- **Dataset:** King County Housing Dataset (Seattle, USA)([Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data))
- **Techniques:**
  - Principal Component Analysis (PCA) for dimensionality reduction
  - Linear Regression on PCA-transformed features
  - R² Score and residual analysis
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Key Outcomes:**
  - Reduced features from 15+ to fewer principal components
  - Balanced model performance with interpretability
  - Visualized actual vs predicted prices

---

## 🧰 Common Stack

- Python 3.10+
- Jupyter Notebooks / Google Colab
- pandas, numpy, matplotlib, seaborn
- scikit-learn, OpenCV, skimage, tensorflow, keras

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Saikeerthansa/ml-dl-projects.git
   cd ml-dl-projects
   ```

2.Install required packages:

```bash
pip install -r requirements.txt
```

3.Then, open any .ipynb file in Jupyter Notebook or Google Colab and run the cells sequentially.
