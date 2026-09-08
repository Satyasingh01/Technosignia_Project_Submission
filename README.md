# Machine Learning & NLP Projects

A collection of machine learning and natural language processing projects focused on solving practical, real-world problems using Python, Scikit-learn, NLP techniques, and modern web technologies.

---

# 1. Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a highly imbalanced, real-world dataset. Two classification models — Logistic Regression and Random Forest — are trained and compared, with evaluation centered on precision and recall rather than plain accuracy.

## Overview

Credit card fraud makes up a tiny fraction of all transactions but causes disproportionate financial harm. This project builds an end-to-end machine learning pipeline — from raw data preprocessing to model training and evaluation — that predicts whether a transaction is legitimate or fraudulent based on anonymized transaction-level features.

**Key result:** Random Forest caught 81 of 98 fraud cases in the test set (**83% recall**) at **94% precision**, outperforming Logistic Regression's 63 of 98 (**64% recall**) at **83% precision**.

## Dataset

* **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Transactions:** 284,807
* **Features:** 31 total — `Time`, `V1`–`V28` (anonymized PCA-transformed features), `Amount`, and target `Class`
* **Class balance:** 492 fraud cases (0.17%) vs. 284,315 legitimate transactions (99.83%)
* **Fraud frequency:** Approximately 1 fraud in every 578 transactions
* **Missing values:** None

## Tools & Libraries

| Library        | Purpose                                            |
| -------------- | -------------------------------------------------- |
| `kagglehub`    | Download dataset directly from Kaggle              |
| `pandas`       | Data loading and manipulation                      |
| `matplotlib`   | Data visualization                                 |
| `seaborn`      | Class imbalance and confusion matrix visualization |
| `scikit-learn` | Preprocessing, model training and evaluation       |

## Project Workflow

1. Import required libraries
2. Download the dataset using `kagglehub`
3. Load the CSV using `pandas`
4. Inspect the dataset using `head()`, `tail()`, `info()`, and `describe()`
5. Check for missing values and data quality issues
6. Analyze class imbalance using `value_counts()`
7. Visualize legitimate vs. fraudulent transactions
8. Split features and target
9. Perform an 80/20 stratified train/test split
10. Drop the `Time` feature
11. Scale the `Amount` feature using `StandardScaler`
12. Train Logistic Regression
13. Train Random Forest
14. Evaluate both models using precision, recall and confusion matrix
15. Compare model performance
16. Select Random Forest as the final model

## Results

| Metric        | Logistic Regression | Random Forest |
| ------------- | ------------------: | ------------: |
| Frauds caught |             63 / 98 |   **81 / 98** |
| Frauds missed |                  35 |        **17** |
| False alarms  |                  13 |         **5** |
| Precision     |                 83% |       **94%** |
| Recall        |                 64% |       **83%** |

Random Forest outperformed Logistic Regression across all major fraud-detection metrics, catching more fraudulent transactions while generating fewer false alarms.

## Why Not Just Use Accuracy?

The dataset contains only **0.17% fraudulent transactions**.

A model that predicts every transaction as legitimate could achieve approximately **99.8% accuracy**, while detecting **zero fraud cases**.

Therefore, this project focuses primarily on:

* **Precision** — How many transactions predicted as fraud were actually fraudulent?
* **Recall** — How many actual fraudulent transactions were successfully detected?
* **Confusion Matrix** — How many fraud cases were caught or missed?

## Limitations

* `V1`–`V28` are PCA-anonymized features, making individual feature interpretation difficult.
* No SMOTE or class-weight balancing was applied.
* Models used default/lightly tuned hyperparameters.
* No cross-validation or extensive hyperparameter optimization was performed.
* Classification threshold remained at the default `0.5`.

## Future Improvements

* Apply SMOTE or class-weight balancing.
* Perform hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
* Evaluate ROC-AUC and Precision-Recall AUC.
* Experiment with XGBoost and LightGBM.
* Perform classification threshold tuning.
* Build a real-time fraud detection API.
* Deploy the final model using FastAPI or Streamlit.

---

# 2. NLP Skill Extractor

An NLP-based web application that automatically extracts technical skills from job descriptions and organizes them into predefined skill categories. The project is designed to help analyze job postings, identify required technologies, and convert unstructured job-description text into structured skill information.

## Overview

Job descriptions contain large amounts of unstructured text, making it difficult to quickly identify the technologies and skills required for a particular role.

This project uses **Natural Language Processing and rule-based skill matching** to identify technical skills from job descriptions and categorize them into a structured skill taxonomy.

The application accepts a job description as input and automatically identifies relevant skills such as:

* Python
* Java
* C/C++
* SQL
* MySQL
* PostgreSQL
* Machine Learning
* Scikit-learn
* TensorFlow
* PyTorch
* AWS
* Azure
* Docker
* Kubernetes
* Git
* React
* Node.js
* NLP
* Generative AI
* LangChain
* RAG
* And many more

## Key Features

### 1. Job Description Input

Users can enter or paste a complete job description into the application.

### 2. Automated Skill Extraction

The NLP pipeline scans the job description and identifies technical skills using a predefined skill taxonomy.

### 3. Skill Normalization

Different names and variations of the same technology are mapped to a common skill.

For example:

| Raw Skill / Variant   | Normalized Skill |
| --------------------- | ---------------- |
| `Python3`             | Python           |
| `Python programming`  | Python           |
| `My SQL`              | MySQL            |
| `Postgres`            | PostgreSQL       |
| `sklearn`             | Scikit-learn     |
| `AWS Cloud`           | AWS              |
| `Amazon Web Services` | AWS              |
| `Py Torch`            | PyTorch          |
| `ReactJS`             | React.js         |
| `NodeJS`              | Node.js          |

### 4. Skill Taxonomy

Extracted skills are organized into categories instead of being returned as an unstructured list.

Example:

```text
Programming
 ├── Python
 ├── Java
 ├── C
 └── JavaScript

Database
 ├── SQL
 ├── MySQL
 ├── PostgreSQL
 └── MongoDB

Machine Learning
 ├── Scikit-learn
 ├── TensorFlow
 ├── PyTorch
 └── Keras

Cloud
 ├── AWS
 ├── Azure
 └── GCP

AI / NLP
 ├── NLP
 ├── LLM
 ├── LangChain
 ├── RAG
 └── Generative AI

DevOps
 ├── Docker
 ├── Kubernetes
 ├── Git
 └── CI/CD
```

### 5. Interactive Web Interface

The project includes a web interface where users can enter job descriptions and view extracted skills in a visually organized format.

The interface includes animated visual elements and a dark-themed UI to make the application more engaging.

## NLP Pipeline

```text
Job Description
       ↓
Text Input / Collection
       ↓
Text Preprocessing
       ↓
Normalization
       ↓
Skill Matching
       ↓
Synonym Resolution
       ↓
Skill Categorization
       ↓
Structured Skill Output
       ↓
Web Application
```

## Skill Extraction Approach

The project uses a **skill taxonomy + synonym mapping + NLP text processing** approach.

Instead of searching only for exact skill names, the system also recognizes common variations and synonyms.

For example:

```text
"Experience with Amazon Web Services"
              ↓
            AWS
```

```text
"Strong knowledge of sklearn"
              ↓
        Scikit-learn
```

```text
"Experience building applications using ReactJS"
              ↓
          React.js
```

This improves extraction consistency across different job descriptions.

## Example

### Input

```text
We are looking for a Data Scientist with experience in
Python, SQL, machine learning, AWS and Power BI.
Knowledge of Scikit-learn, Pandas and NumPy is preferred.
```

### Extracted Skills

```text
Programming:
    Python

Database:
    SQL

Machine Learning:
    Machine Learning
    Scikit-learn

Data Analytics:
    Pandas
    NumPy
    Power BI

Cloud:
    AWS
```

## Technologies Used

| Technology          | Purpose                          |
| ------------------- | -------------------------------- |
| Python              | Core programming language        |
| Pandas              | Data processing                  |
| NumPy               | Numerical operations             |
| NLP                 | Job-description text processing  |
| Regular Expressions | Pattern and skill matching       |
| Scikit-learn        | ML/NLP utilities                 |
| HTML                | Application structure            |
| CSS                 | UI styling and animations        |
| JavaScript          | Frontend interactions            |
| Jupyter Notebook    | Data exploration and development |

## Project Workflow

1. Collect job-description data
2. Load the dataset using Pandas
3. Perform exploratory data analysis
4. Inspect job-description quality
5. Clean and normalize text
6. Create a structured skill taxonomy
7. Define skill synonyms and variations
8. Extract skills from job descriptions
9. Normalize extracted skills
10. Categorize skills
11. Validate extracted skills against job descriptions
12. Build an interactive web interface
13. Display extracted skills in categorized sections

## Applications

The Skill Extractor can be used for:

* Job market analysis
* Resume optimization
* Job-to-skill matching
* Recruitment automation
* Skill-gap analysis
* Candidate screening
* Identifying frequently requested technologies
* Building job recommendation systems
* Analyzing technology trends across job postings

## Future Improvements

* Implement Named Entity Recognition (NER) for improved skill identification.
* Use transformer-based NLP models such as BERT.
* Add semantic similarity using embeddings.
* Implement vector databases for skill matching.
* Add resume skill extraction.
* Build job-to-resume matching.
* Calculate candidate skill-match scores.
* Add skill-gap recommendations.
* Support multiple languages.
* Build a REST API using FastAPI.
* Deploy the application using Docker and cloud services.
* Add an LLM-based extraction pipeline for ambiguous skills.

---

# Projects Comparison

| Project                     | Domain               | Problem Solved                                 | Key Techniques                                                      |
| --------------------------- | -------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| Credit Card Fraud Detection | Machine Learning     | Detect fraudulent transactions                 | Classification, Feature Scaling, Random Forest, Logistic Regression |
| NLP Skill Extractor         | NLP / Data Analytics | Extract technical skills from job descriptions | NLP, Regex, Taxonomy, Synonym Mapping, Text Processing              |

---

# Skills Demonstrated

Through these projects, the following practical skills were demonstrated:

**Programming:**
Python, SQL

**Machine Learning:**
Classification, Logistic Regression, Random Forest, Model Evaluation, Feature Scaling

**NLP:**
Text Preprocessing, Skill Extraction, Entity Recognition Concepts, Taxonomy Design, Synonym Mapping

**Data Analysis:**
Pandas, NumPy, Exploratory Data Analysis, Data Cleaning

**Visualization:**
Matplotlib, Seaborn

**Web Development:**
HTML, CSS, JavaScript

**Tools:**
Jupyter Notebook, Kaggle, Git/GitHub

---

# Author

**Satyam Singh**
Data Analytics | Machine Learning | NLP | Generative AI
