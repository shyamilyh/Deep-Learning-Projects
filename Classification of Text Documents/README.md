# 📚 Classification of Text Documents  

### 📘 Overview  
This project demonstrates an **end-to-end text classification pipeline** — from data preprocessing to model evaluation — using **TF-IDF features** and multiple machine learning classifiers.  
The objective is to classify text documents into predefined categories and compare model performance to identify the best-performing algorithm.  

---

## ⚙️ Project Workflow  

### 1️⃣ Data Loading  
- Loaded dataset using `pandas.read_csv("synthetic_text_data.csv")`.
- Inspected structure, columns, and ensured there were no missing values.
- The dataset contains:
  - **text** → document text  
  - **label** → category/class of the text

---

### 2️⃣ Text Preprocessing  
Used **NLTK** for text cleaning and normalization. The pipeline includes:
- Converting text to lowercase  
- Removing punctuation and special characters (via regex)  
- Tokenizing words using `word_tokenize()`  
- Removing English stopwords (`nltk.corpus.stopwords`)  
- Joining tokens back to form cleaned text  

A new column `preprocessed_text` was created to store cleaned text for model input.

---

### 3️⃣ Feature Extraction — TF-IDF Vectorization  
- Used `TfidfVectorizer(max_features=5000)` to convert preprocessed text into numerical features.
- Captures term importance relative to the entire corpus (Term Frequency–Inverse Document Frequency).
- Resulting feature matrix is sparse but efficient for text-based models.

---

### 4️⃣ Data Splitting  
- Performed an **80/20 split** using `train_test_split()` from scikit-learn.  
- Training set → used to train models.  
- Test set → used to evaluate model performance on unseen data.

---

### 5️⃣ Model Training  
Trained and compared four machine learning classifiers:

| Model | Description |
|--------|--------------|
| **Multinomial Naive Bayes** | Probabilistic baseline model suited for word counts and TF-IDF data. |
| **Support Vector Machine (SVC)** | Captures complex decision boundaries between classes. |
| **Logistic Regression** | Linear classifier, efficient for large sparse datasets. |
| **MLP Classifier (Neural Network)** | Nonlinear model capable of capturing feature interactions. |

Each model was trained using `fit(X_train, y_train)` on the TF-IDF feature set.

---

### 6️⃣ Model Evaluation  
Models were evaluated using standard classification metrics:  
- **Accuracy**  
- **Precision, Recall, and F1-Score** via `classification_report()`  

**Results (from notebook):**  

| Model | Accuracy |
|--------|-----------|
| Multinomial Naive Bayes | 0.6818 |
| Support Vector Machine | 0.6364 |
| Logistic Regression | 0.6818 |
| **MLP Classifier** | **0.8636** ✅ |

**Best Model:** MLP Classifier (highest accuracy and balanced metrics)

---

### 7️⃣ Interpretation & Insights  
- The **MLP Classifier** outperformed other traditional models, likely due to its nonlinear feature interactions.  
- **Naive Bayes** and **Logistic Regression** showed similar performance (~68%).  
- **SVM** lagged slightly behind due to lack of kernel tuning.  
- TF-IDF with 5000 features captured enough lexical variation for good accuracy while avoiding overfitting.

---

## 🧠 Key Learnings  

1. **TF-IDF** remains a strong baseline for text classification.  
2. Simple neural models (MLP) can outperform linear classifiers even on sparse text data.  
3. Proper text cleaning (tokenization, stopword removal) greatly improves performance.  
4. Model performance can further improve with **cross-validation** and **hyperparameter tuning**.  

---

## ⚙️ Future Improvements  

- 🔹 Implement **GridSearchCV** to optimize hyperparameters (e.g., C for SVM, alpha for Naive Bayes).  
- 🔹 Use **Pipelines** to integrate preprocessing, vectorization, and modeling into one reproducible workflow.  
- 🔹 Explore **n-grams** and **word embeddings** (Word2Vec, GloVe, or BERT).  
- 🔹 Apply **cross-validation** for more reliable generalization estimates.  
- 🔹 Deploy model using Flask or Streamlit for real-time classification.  

---

## 🧰 Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python |
| Libraries | Pandas, NumPy, NLTK, Scikit-learn |
| Models | Naive Bayes, SVM, Logistic Regression, MLP Classifier |
| Feature Extraction | TF-IDF Vectorizer |
| Environment | Jupyter Notebook |

---

## ✅ Results & Conclusion  

The notebook successfully demonstrates a **complete text classification pipeline** with clear comparisons among multiple models.  
The **MLP Classifier** achieved the best performance with **86.36% accuracy**, showing the power of even simple neural architectures for textual data.

This project provides a robust foundation for natural language processing tasks such as document categorization, spam detection, or sentiment analysis.


