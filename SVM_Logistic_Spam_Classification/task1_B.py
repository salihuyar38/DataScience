import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

file_path = "SMSSpamCollection"
data = pd.read_csv(file_path, sep="\t", names=["label", "text"], header=None)

data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['text']
y = data['label']

def create_pipeline(kernel='linear', C=1.0):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5)),
        ('svm', SVC(kernel=kernel, C=C))
    ])
    return pipeline

kernels = ['linear', 'rbf']
C_values = [0.1, 1, 10]
results = []

for kernel in kernels:
    for C in C_values:
        pipeline = create_pipeline(kernel=kernel, C=C)
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        avg_score = scores.mean()
        results.append({'Kernel': kernel, 'C': C, 'Accuracy': avg_score})

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8, 6))
for kernel in kernels:
    subset = results_df[results_df['Kernel'] == kernel]
    plt.plot(subset['C'], subset['Accuracy'], marker='o', label=f'Kernel: {kernel}')

plt.xscale('log')
plt.xlabel("C (Log Scale)")
plt.ylabel("Accuracy")
plt.title("SVM Cross-Validation Results")
plt.legend()
plt.grid()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)

best_C = 1
best_kernel = 'linear'
final_pipeline = create_pipeline(kernel=best_kernel, C=best_C)
final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
