from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def to_libsvm(df, output_file):
    """Convert a DataFrame to LibSVM format."""
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            label = row["label"]
            features = row.drop("label")
            feature_str = " ".join([f"{i+1}:{value}" for i, value in enumerate(features) if value != 0])
            f.write(f"{label} {feature_str}\n")

def main():
  
    print("\n[1] Fetching the Spambase dataset...")
    spambase = fetch_ucirepo(id=94)

    
    X = spambase.data.features
    y = spambase.data.targets.astype(int).values.ravel()  
   
    print("[2] Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)

   
    print("[3] Saving datasets to LibSVM format...")
    train_df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name="label")], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name="label")], axis=1)
    to_libsvm(train_df, "train.libsvm")
    to_libsvm(test_df, "test.libsvm")

    
    print("\n[4] Training and evaluating LinearSVC...")
    C_values = [0.1, 1, 10, 100]
    results = []

    for C in C_values:
        print(f"  - Training LinearSVC with C={C}...")
        clf = LinearSVC(C=C, max_iter=5000, random_state=31)
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)  
        avg_acc = scores.mean()
        results.append((C, avg_acc))
        print(f"    Accuracy: {avg_acc:.4f}")

  
    print("\n[5] Plotting results...")
    C_vals, accuracies = zip(*results)
    plt.plot(C_vals, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel("C (Log Scale)")
    plt.ylabel("Accuracy")
    plt.title("LinearSVC Hyperparameter Optimization")
    plt.grid(True)
    plt.show()

   
    best_C = max(results, key=lambda x: x[1])[0]
    print(f"\n[6] Training final model with C={best_C}...")
    best_clf = LinearSVC(C=best_C, max_iter=5000, random_state=31)
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
