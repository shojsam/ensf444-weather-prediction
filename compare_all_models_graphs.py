import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("weather_classification_data.csv")
METRICS_FIGURE_PATH = Path("model_comparison_metrics.png")
CONFUSION_FIGURE_PATH = Path("model_comparison_confusion_matrices.png")
RANDOM_STATE = 42


def build_preprocessor(numeric_columns, categorical_columns, scale_numeric):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_columns),
            ("cat", Pipeline(steps=categorical_steps), categorical_columns),
        ]
    )


def make_models(numeric_columns, categorical_columns):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                        scale_numeric=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                        scale_numeric=False,
                    ),
                ),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                        scale_numeric=False,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def plot_metric_comparison(results_df, class_labels):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    accuracy_columns = ["validation_accuracy", "test_accuracy", "cv_mean_accuracy"]
    results_df.plot(
        x="model",
        y=accuracy_columns,
        kind="bar",
        ax=axes[0],
        rot=0,
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylabel("Score")
    axes[0].legend(["Validation", "Test", "CV Mean"])

    f1_data = pd.DataFrame(
        {
            label: [report[label]["f1-score"] for report in results_df["classification_report"]]
            for label in class_labels
        },
        index=results_df["model"],
    )
    f1_data.plot(
        kind="bar",
        ax=axes[1],
        rot=0,
        color=["#E45756", "#72B7B2", "#B279A2", "#FF9DA6"],
    )
    axes[1].set_ylim(0.75, 1.0)
    axes[1].set_title("Per-Class F1 Score Comparison")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(title="Weather Type")

    fig.tight_layout()
    fig.savefig(METRICS_FIGURE_PATH, dpi=300, bbox_inches="tight")


def plot_confusion_matrices(results_df, class_labels):
    plt.style.use("default")
    fig, axes = plt.subplots(1, len(results_df), figsize=(18, 5), constrained_layout=True)

    for axis, (_, row) in zip(axes, results_df.iterrows()):
        matrix = row["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(row["model"])
        axis.set_xticks(np.arange(len(class_labels)))
        axis.set_yticks(np.arange(len(class_labels)))
        axis.set_xticklabels(class_labels, rotation=45, ha="right")
        axis.set_yticklabels(class_labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
    fig.savefig(CONFUSION_FIGURE_PATH, dpi=300, bbox_inches="tight")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Weather Type"])
    y = df["Weather Type"]

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    class_labels = list(target_encoder.classes_)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    models = make_models(numeric_columns, categorical_columns)
    results = []

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=1)

        results.append(
            {
                "model": model_name,
                "validation_accuracy": accuracy_score(y_val, y_val_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std_accuracy": cv_scores.std(),
                "classification_report": classification_report(
                    y_test,
                    y_test_pred,
                    target_names=class_labels,
                    output_dict=True,
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    y_test_pred,
                    labels=range(len(class_labels)),
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)

    summary = results_df[["model", "validation_accuracy", "test_accuracy", "cv_mean_accuracy", "cv_std_accuracy"]]
    print("\nModel performance summary\n")
    print(summary.to_string(index=False))

    print("\nPer-class test F1 scores\n")
    f1_summary = pd.DataFrame(
        {
            "model": results_df["model"],
            **{
                label: results_df["classification_report"].apply(lambda report: report[label]["f1-score"])
                for label in class_labels
            },
        }
    )
    print(f1_summary.to_string(index=False))

    plot_metric_comparison(results_df, class_labels)
    plot_confusion_matrices(results_df, class_labels)

    print(f"\nSaved graph image: {METRICS_FIGURE_PATH}")
    print(f"Saved graph image: {CONFUSION_FIGURE_PATH}")


if __name__ == "__main__":
    main()
