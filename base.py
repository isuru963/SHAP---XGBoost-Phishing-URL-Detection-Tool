!pip install xgboost shap --quiet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
df = pd.read_csv("/content/top30_features_dataset.csv")
df.head()
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})
X = df.drop(columns=["url", "status"])
y = df["status"]
print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

# Prediction probabilities
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# AUC score
auc_score = roc_auc_score(y_test, y_prob)
print("ROC–AUC Score:", round(auc_score, 4))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")

shap.summary_plot(shap_values, X_test)

shap_importance = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_Importance": shap_importance
}).sort_values(by="SHAP_Importance", ascending=False)

feature_importance.head(15)

feature_descriptions = {
    "google_index": "the website is not indexed by Google",
    "page_rank": "the website has a very low page rank",
    "nb_www": "the URL contains multiple 'www' tokens",
    "nb_hyperlinks": "the webpage contains an unusually high number of hyperlinks",
    "phish_hints": "the URL contains phishing-related keywords",
    "domain_age": "the domain is very new or recently registered",
    "web_traffic": "the website has very low web traffic",
    "nb_hyphens": "the URL contains many hyphens",
    "nb_slash": "the URL contains an excessive number of slashes",
    "length_hostname": "the hostname length is unusually long",
    "nb_dots": "the URL contains a high number of dots",
    "longest_word_path": "the URL path contains an unusually long word",
    "ratio_digits_url": "the URL contains a high ratio of digits",
    "ratio_extHyperlinks": "the page contains many external hyperlinks",
    "safe_anchor": "the webpage uses unsafe anchor text links",
    "domain_in_title": "the domain name does not appear in the page title",
    "length_url": "the overall URL length is unusually long",
    "https_token": "the URL misuses the HTTPS token",
    "ratio_digits_host": "the hostname contains many numeric characters",
    "links_in_tags": "suspicious links are embedded inside HTML tags"
}

def explain_prediction(index, top_k=3):
    pred = xgb_model.predict(X_test.iloc[[index]])[0]
    label = "Phishing" if pred == 1 else "Legitimate"

    shap_sample = shap_values[index]
    top_features = np.argsort(np.abs(shap_sample))[-top_k:][::-1]

    reasons = []
    for i in top_features:
        feature = X.columns[i]
        direction = "increases" if shap_sample[i] > 0 else "decreases"
        description = feature_descriptions.get(feature, feature)
        reasons.append(f"{description} which {direction} the phishing likelihood")

    return (
        f"This URL is classified as {label} because "
        + ", and ".join(reasons)
        + "."
    )

print(explain_prediction(index=0))

results = []

for i in range(len(X_test)):
    results.append({
        "URL": df.loc[X_test.index[i], "url"],
        "Prediction": "Phishing" if xgb_model.predict(X_test.iloc[[i]])[0] == 1 else "Legitimate",
        "Explanation": explain_prediction(i)
    })

explainable_results_df = pd.DataFrame(results)
explainable_results_df.head(10)
