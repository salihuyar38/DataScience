from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

file_path = "SMSSpamCollection"
data = pd.read_csv(file_path, sep="\t", names=["label", "text"], header=None)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['text']
y = data['label']

def create_pipeline(model, **kwargs):
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5)),
        ('classifier', model(**kwargs))
    ])
def compare_models():
    models = {
        "SVM": SVC,
        "Logistic Regression": LogisticRegression
    }
    hyperparameters = {
        "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear']},
        "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    }
    results = []
    for model_name, model in models.items():
        for params in hyperparameters[model_name]:
            for value in hyperparameters[model_name][params]:
                pipeline = create_pipeline(model, **{params: value})
                scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
                avg_score = scores.mean()
                results.append({
                    'Model': model_name,
                    'C': value,
                    'Accuracy': avg_score,
                    'kernel': 'linear' if model_name == 'SVM' else None,
                    'solver': 'liblinear' if model_name == 'Logistic Regression' else None
                })
    results_df = pd.DataFrame(results)
    return results_df

results_df = compare_models()
print("\nComparison Results:")
print(results_df)

plt.figure(figsize=(10, 6))
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    subset = subset[pd.to_numeric(subset['C'], errors='coerce').notnull()]
    plt.plot(subset['C'], subset['Accuracy'], marker='o', label=model)

plt.xscale('log')
plt.xlabel("C (Log Scale)")
plt.ylabel("Accuracy")
plt.title("Comparison of SVM and Logistic Regression")
plt.legend()
plt.grid()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_C = 1
best_kernel = 'linear'
final_svm_pipeline = create_pipeline(SVC, C=best_C, kernel=best_kernel)
final_svm_pipeline.fit(X_train, y_train)
y_pred_svm = final_svm_pipeline.predict(X_test)
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

best_solver = 'liblinear'
final_lr_pipeline = create_pipeline(LogisticRegression, C=best_C, solver=best_solver)
final_lr_pipeline.fit(X_train, y_train)
y_pred_lr = final_lr_pipeline.predict(X_test)
print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
