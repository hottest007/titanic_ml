# titanic_ml
A titanic survival prediction machine learning model using logistic regression

# 🚢 Titanic Survival Prediction System

## 📌 Project Overview

This project builds a complete machine learning system to predict passenger survival on the Titanic dataset. It goes beyond a simple model by incorporating:

* Data preprocessing and feature engineering
* A reproducible training pipeline
* Model persistence (saving/loading)
* A production-ready API using FastAPI
* An interactive user interface using Gradio

The goal is to demonstrate a **clean, end-to-end ML workflow** following professional best practices.

---

## 🎯 Objective

The objective of this project is to build a **balanced and reliable classification model** that predicts whether a passenger survived or not.

Rather than aggressively optimizing for a single metric (e.g., recall or precision), the model is designed to:

> ✅ Maintain a **balanced performance across both classes**
> ✅ Provide consistent and interpretable predictions
> ✅ Avoid overfitting or bias toward one outcome

---

## 🧠 Modeling Approach

This project uses **Logistic Regression** as the final model.

### Why Logistic Regression?

* Performs well on structured/tabular data
* Produces stable and interpretable results
* Less prone to overfitting compared to complex models
* Provides a good balance between precision and recall

---

## ⚙️ Data Processing Pipeline

The model is built using a **scikit-learn Pipeline**, ensuring all steps are reproducible and consistent.

### 🔹 Feature Engineering

* Extracted **Title** from passenger names (e.g., Mr, Mrs, Miss)
* Extracted **Deck** from cabin information
* Handled missing values:

  * Age → filled with median
  * Fare → filled with median
  * Deck → filled with "Missing"

### 🔹 Preprocessing

* Numerical features:

  * Imputed using median
* Categorical features:

  * Imputed using most frequent value
  * Encoded using One-Hot Encoding

---

## 🤖 Model Pipeline

The final pipeline consists of:

1. **ColumnTransformer**

   * Applies different preprocessing to numeric and categorical features
2. **Logistic Regression Model**

   * Configured with:

     * `max_iter=1000`
     * `class_weight="balanced"`

This ensures the model handles class imbalance without overcompensating.

---

## 📊 Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

### ✅ Key Design Decision

The model is intentionally kept **balanced**, meaning:

* It does not favor predicting survival over non-survival
* It does not aggressively maximize recall at the expense of precision
* It maintains fair performance across both classes

This reflects a **general-purpose classification system** rather than a specialized or biased one.

---

## 🏗️ Project Structure

```text
titanic_project/
│
├── data/
│   └── Titanic-Dataset.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   └── train.py
│   │
│   └── serving/
│       └── inference.py
│
├── model/
│   └── titanic_logreg.pkl
│
├── app/
│   └── app.py
│
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train the model

```bash
python src/models/train.py
```

This will:

* Train the pipeline
* Save the model to `model/titanic_logreg.pkl`

---

### 3. Start the API server

```bash
uvicorn app.app:app --reload
```

---

## 🌐 Access the Application

* **FastAPI Docs:**
  http://127.0.0.1:8000/docs

* **Gradio UI:**
  http://127.0.0.1:8000/ui

---

## 🔌 API Endpoint

### POST `/predict`

#### Example Request:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S",
  "Title": "Mr",
  "Deck": "Missing"
}
```

#### Example Response:

```json
{
  "prediction": 0
}
```

* `0` → Did Not Survive
* `1` → Survived

---

## 🖥️ Gradio Interface

A simple web UI is included to:

* Input passenger details
* Get instant predictions
* Demonstrate the model to non-technical users

---

## 🧠 Key Design Principles

* **Separation of concerns**

  * Data loading, feature engineering, training, and inference are modularized

* **Reproducibility**

  * Entire pipeline is saved and reused

* **Consistency**

  * Same preprocessing used in training and inference

* **Simplicity**

  * Avoided unnecessary complexity and over-tuning

---

## 🧭 Future Improvements

* Add probability outputs (confidence scores)
* Implement threshold tuning for flexible decision-making
* Add model explainability (e.g., SHAP)
* Deploy to cloud (Render, AWS, etc.)
* Add logging and monitoring

---

## 👨‍💻 Author Notes

This project focuses on building a **clean and production-ready ML system**, rather than chasing marginal performance gains.

The decision to keep the model balanced reflects an understanding that:

> A well-generalized and interpretable model is often more valuable than a highly tuned but unstable one.

---

## 📜 License

This project is open-source and available for learning and demonstration purposes.
