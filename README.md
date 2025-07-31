# 🧠 Natural Language Processing with Disaster Tweets (Kaggle)

This repository contains my full pipeline for the Kaggle competition [NLP - Getting Started](https://www.kaggle.com/competitions/nlp-getting-started). The goal is to classify whether a tweet is about a real disaster or not (`target = 1` or `0`).

### 🚀 Installation

1. Clone the repository:  
```  
git clone https://github.com/SergKhachikyan/London_House_Price_Prediction_Advanced_Techniques.git
```
2. Change directory:
```
cd London_House_Price_Prediction_Advanced_Techniques
```
3. Create virtual environment:
```
py -m venv venv
```
4. Activate virtual environment:
```
venv\Scripts\activate
```
5. Update the package manager:
```
py -m pip install -U pip
```
6.Install dependencies:
```
pip install -r requirements.txt  
```
7.Launch the notebook:
```
jupyter notebook untitled.ipynb
```  


### 📂 Project Structure:
- `train.csv` / `test.csv` – original dataset
- `submission.csv` – prediction file for Kaggle submission
- `NLP_Disaster_Tweets.ipynb` – main notebook with full preprocessing, training and evaluation
- `README.md` – this file

### ✅ Key Steps:
- Missing values in `keyword` filled using weighted random sampling based on value_counts distribution (with `normalize=True`)
- Texts cleaned with `clean_text()` function: lowercased, removed links, punctuation, digits, extra whitespaces
- `TfidfVectorizer` used with `max_features=10000`, `ngram_range=(1,2)`, `stop_words='english'`, `min_df=3`
- Trained multiple models: `RandomForestClassifier`, `GridSearchCV(RF)`, `XGBoostClassifier`, `LogisticRegression(class_weight='balanced')`
- Best performance: `LogisticRegression` with TF-IDF vectorizer gave **Public Score: `0.79098`**
- Evaluation based on `classification_report()` with attention to F1-score for class `1` (disaster-related tweets)

### 📊 Model Comparison (Validation Set):

| Model                       | Accuracy | F1-Score (1) | Macro Avg | Public Score |
|----------------------------|----------|--------------|-----------|--------------|
| RandomForest (basic)       | 0.71     | 0.49         | 0.64      | 0.75789      |
| RandomForest (tuned)       | 0.76     | 0.68         | 0.74      | 0.76677      |
| XGBoostClassifier          | 0.77     | 0.69         | 0.75      | 0.75789      |
| LogisticRegression (best)  | 0.76     | 0.68–0.70    | 0.74–0.76 | **0.79098 ✅** |

### 🔥 Best Model:

LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

### 🛠 Possible Next Steps:

- Include keyword as encoded feature
- Try StackingClassifier with LogReg + XGB + RF
- Switch to transformers like BERT/RoBERTa
- Tune TF-IDF (min_df, max_df) or try CountVectorizer
- Feature importance via coef_ in LogReg

### 📈 Submission Format:
csv
id,target
0,1
1,0
2,1
...

### 💬 Final Note:
This repo helped me go from 0.75x public score to 0.79098. The most effective combo was cleaned text + tf-idf + balanced logistic regression. Clean, simple, and powerful.
