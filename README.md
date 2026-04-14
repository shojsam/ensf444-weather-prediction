# ENSF 444 – Weather Type Classification

A machine learning project that classifies weather conditions (**Sunny, Cloudy, Rainy, Snowy**) from atmospheric sensor data using three models: **Logistic Regression**, **Decision Tree**, and **Random Forest**.

**Repository:** https://github.com/shojsam/ensf444-weather-prediction.git

---

## Setup

```bash
git clone https://github.com/shojsam/ensf444-weather-prediction.git
cd ensf444-weather-prediction
pip install pandas numpy matplotlib scikit-learn notebook
```

---

## How to Run

### Step 1 — Run the Individual Model Notebooks

Each notebook trains and evaluates a single model. Open them in Jupyter to see each model's accuracy, classification report, confusion matrix, and feature importances.

```bash
jupyter notebook
```

Open each notebook and run all cells (`Kernel → Restart & Run All`):

| Notebook | Model | Expected Accuracy |
|----------|-------|------------------|
| `Logistic_Regression_Weather_Type_Classification.ipynb` | Logistic Regression (linear baseline) | ~88% |
| `model_using_decisiontrees.ipynb` | Decision Tree (non-linear) | ~88% |
| `random_forest_weather_pipeline.ipynb` | Random Forest (non-linear ensemble) | ~95% |

> Make sure `weather_classification_data.csv` is in the same folder as the notebooks before running.

---

### Step 2 — Compare All Models Using the Python Script

If you want to compare all three models side by side, run the comparison script. It trains all models on the same data split and saves the results as charts.

```bash
python compare_all_models_graphs.py
```

**Expected console output:**

```
Model performance summary

            model  validation_accuracy  test_accuracy  cv_mean_accuracy
    Random Forest               ~0.94         ~0.95            ~0.94
    Decision Tree               ~0.88         ~0.88            ~0.88
Logistic Regression             ~0.88         ~0.88            ~0.88
```

**Chart files saved to the same folder:**
- `model_comparison_metrics.png` – Accuracy & F1 score bar charts
- `model_comparison_confusion_matrices.png` – Side-by-side confusion matrices
- `final_unseen_test_accuracy.png` – Final test accuracy per model
- `random_forest_per_class_accuracy.png` – Random Forest per-class accuracy

---

## Conclusion

**🏆 Random Forest is the best-performing model**, achieving approximately **~95% test accuracy** — around 7 percentage points higher than both Logistic Regression and Decision Tree (~88%).

Random Forest wins because it aggregates 300 decision trees (bagging + feature randomization), which reduces overfitting and captures the non-linear relationships between features like temperature, humidity, pressure, and season that simpler models cannot handle as effectively.

| Model | Test Accuracy |
|-------|--------------|
| **Random Forest** | **~95%** |
| Decision Tree | ~88% |
| Logistic Regression | ~88% |

---

## References

1. Scikit-learn: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Dataset: [Weather Type Classification – Kaggle](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)
4. **AI Tools:** Google Gemini and ChatGPT were used to assist with understanding ML concepts (Gini impurity, bagging, cross-validation) and for guidance on structuring scikit-learn pipelines. 
