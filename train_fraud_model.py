# Full notebook cell: read CSV, clean, train multiple models, show results and accuracy summary
#matplotlib inline
import io
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report,
                             precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# reproducibility
RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

# --------------- USER CONFIG ---------------
csv_path = r'C:\Users\hp\OneDrive\Desktop\FYP\server\Fraud.csv'  # <-- set to your CSV path (or None)
# -------------------------------------------

# If you want to paste a fallback CSV text into csv_text variable, you can.
# If not provided, we'll create a tiny synthetic fallback to avoid errors.
try:
    csv_text
except NameError:
    csv_text = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n1,TRANSFER,1000,C123,1000,0,D456,0,1000,0,0\n2,CASH_OUT,500,C789,500,0,D012,100,600,1,0\n"

# ---------------- Read CSV ----------------
if csv_path and os.path.exists(csv_path):
    print(f"Reading CSV from path: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    print("No csv_path provided or file not found - using fallback CSV text sample.")
    df = pd.read_csv(io.StringIO(csv_text))

print("Initial rows:", len(df))
print("Columns:", df.columns.tolist())

# ---------------- Cleaning ----------------
expected_cols = ['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig',
                 'nameDest','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Convert numeric columns where possible
num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing critical numeric fields
before = len(df)
df = df.dropna(subset=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud'])
after = len(df)
print(f"Dropped {before-after} rows with missing critical numeric values.")

# Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Dropped {before-after} exact duplicate rows.")

print("Rows after cleaning:", len(df))
print("Fraud counts (isFraud):")
print(df['isFraud'].value_counts())

# ---------------- Feature engineering ----------------
df['orig_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

df['orig_became_zero'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
df['dest_became_zero'] = ((df['newbalanceDest'] == 0) & (df['oldbalanceDest'] > 0)).astype(int)

df = pd.get_dummies(df, columns=['type'], prefix='type')

# drop identifiers
df = df.drop(columns=['nameOrig','nameDest'])

feature_cols = [c for c in df.columns if c not in ['isFraud','isFlaggedFraud']]
print("Feature columns (example):", feature_cols[:20])

# ---------------- Prepare X, y ----------------
X = df[feature_cols].copy()
y = df['isFraud'].astype(int).copy()

# Train-test split
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RND, stratify=y)
print("Train shape:", X_train_df.shape, "Test shape:", X_test_df.shape)

# Scale numeric features (fit on training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

# Apply SMOTE to balance training data
smote = SMOTE(random_state=RND)
print("Applying SMOTE to training set...")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE ->", X_train_resampled.shape, "frauds in train:", int(y_train_resampled.sum()))

# ---------------- Training / Models ----------------
def train_all_models(X_train, X_test, y_train, y_test):
    results = {}

    # Logistic Regression
    print("\nTraining: Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=RND)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    results['Logistic Regression'] = {
        'model': lr,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, zero_division=0),
        'recall': recall_score(y_test, lr_pred, zero_division=0),
        'f1': f1_score(y_test, lr_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, lr_pred),
        'feature_importance': np.abs(lr.coef_[0])
    }

    print("Done: Logistic Regression -> F1:", results['Logistic Regression']['f1'])

    # Random Forest
    print("\nTraining: Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RND,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    results['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, rf_pred),
        'feature_importance': rf.feature_importances_
    }

    print("Done: Random Forest -> F1:", results['Random Forest']['f1'])

    # XGBoost
    print("\nTraining: XGBoost")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RND, eval_metric='logloss', use_label_encoder=False)
    scale_pos_weight = max(1.0, (len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))))
    xgb.set_params(scale_pos_weight=scale_pos_weight)
    xgb.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'model': xgb, 'predictions': xgb_pred, 'probabilities': xgb_proba,
        'accuracy': accuracy_score(y_test, xgb_pred), 'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0), 'f1': f1_score(y_test, xgb_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred), 'feature_importance': xgb.feature_importances_ if hasattr(xgb, 'feature_importances_') else None
    }
    print("Done: XGBoost -> F1:", results['XGBoost']['f1'])

    # RNN (LSTM) - expects numpy arrays shaped (n_samples, features)
    print("\nTraining: RNN (LSTM) - this may take some time")
    try:
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    except Exception:
        # Fallback: convert to numpy arrays (shouldn't be necessary but safe)
        X_train_arr = np.array(X_train)
        X_test_arr = np.array(X_test)
        X_train_lstm = X_train_arr.reshape((X_train_arr.shape[0], 1, X_train_arr.shape[1]))
        X_test_lstm = X_test_arr.reshape((X_test_arr.shape[0], 1, X_test_arr.shape[1]))

    rnn = Sequential([
        LSTM(64, input_shape=(1, X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    class_weight = {0: 1.0, 1: float(scale_pos_weight)}
    rnn.fit(X_train_lstm, y_train, epochs=10, batch_size=128, validation_split=0.15, verbose=1,
            callbacks=[early_stop], class_weight=class_weight)
    rnn_proba = rnn.predict(X_test_lstm, verbose=0).flatten()
    rnn_pred = (rnn_proba > 0.5).astype(int)
    results['RNN (LSTM)'] = {
        'model': rnn, 'predictions': rnn_pred, 'probabilities': rnn_proba,
        'accuracy': accuracy_score(y_test, rnn_pred), 'precision': precision_score(y_test, rnn_pred, zero_division=0),
        'recall': recall_score(y_test, rnn_pred, zero_division=0), 'f1': f1_score(y_test, rnn_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, rnn_pred), 'feature_importance': None
    }
    print("Done: RNN (LSTM) -> F1:", results['RNN (LSTM)']['f1'])

    # save LSTM model
    rnn.save("lstm_fraud_model")

    # save scaler
    
    joblib.dump(scaler, "scaler.pkl")

    print("LSTM model and scaler saved successfully")


    # Voting
    print("\nTraining: Ensemble (Voting)")
    voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft', n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    voting_pred = voting_clf.predict(X_test)
    voting_proba = voting_clf.predict_proba(X_test)[:, 1]
    results['Ensemble (Voting)'] = {
        'model': voting_clf, 'predictions': voting_pred, 'probabilities': voting_proba,
        'accuracy': accuracy_score(y_test, voting_pred), 'precision': precision_score(y_test, voting_pred, zero_division=0),
        'recall': recall_score(y_test, voting_pred, zero_division=0), 'f1': f1_score(y_test, voting_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, voting_pred), 'feature_importance': None
    }
    print("Done: Voting -> F1:", results['Ensemble (Voting)']['f1'])

    #Stacking 
    print("\nTraining: Ensemble (Stacking)")

    stacking_clf = StackingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=RND,
            n_jobs=1
        ),
        cv=3,                 
        n_jobs=1,             
        passthrough=False    
    )

    stacking_clf.fit(X_train, y_train)

    stacking_pred = stacking_clf.predict(X_test)
    stacking_proba = stacking_clf.predict_proba(X_test)[:, 1]

    results['Ensemble (Stacking)'] = {
        'model': stacking_clf,
        'predictions': stacking_pred,
        'probabilities': stacking_proba,
        'accuracy': accuracy_score(y_test, stacking_pred),
        'precision': precision_score(y_test, stacking_pred, zero_division=0),
        'recall': recall_score(y_test, stacking_pred, zero_division=0),
        'f1': f1_score(y_test, stacking_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, stacking_pred),
        'feature_importance': None
    }

    print("Done: Stacking -> F1:", results['Ensemble (Stacking)']['f1'])

# Execute training (passes scaled + resampled training arrays, and scaled test arrays)
t0 = time.time()
results = train_all_models(X_train_resampled, X_test_scaled, y_train_resampled, y_test)
t1 = time.time()
print(f"\nAll models trained in {(t1-t0):.1f} seconds (wall-clock)")

# ---------------- Reporting ----------------
def generate_csv_report(results, filename='model_comparison_report.csv'):
    report_data = []
    for model_name, result in results.items():
        cm = result['confusion_matrix']
        tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        report_data.append({
            'Model': model_name, 'Accuracy': result['accuracy'], 'Precision': result['precision'],
            'Recall': result['recall'], 'F1-Score': result['f1'], 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })
    df_report = pd.DataFrame(report_data).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    df_report.to_csv(filename, index=False)
    print(f"Saved CSV report -> {filename}")
    return df_report

df_report = generate_csv_report(results)
display(df_report)

# Print classification reports
for name, res in results.items():
    print("\n" + "="*60)
    print("Model:", name)
    print(classification_report(y_test, res['predictions'], digits=4, zero_division=0))

# Confusion matrices (matplotlib)
def plot_confusion_matrices(results):
    models = list(results.keys())
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    for idx, model_name in enumerate(models):
        cm = results[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
        axes[idx].set_title(f"{model_name}\nAcc: {results[model_name]['accuracy']:.4f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(results)

# ROC curves (plotly)
def plot_roc_curves(results):
    fig = go.Figure()
    for model_name, result in results.items():
        try:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC={roc_auc:.4f})', mode='lines'))
        except Exception:
            pass
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', mode='lines', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR', width=900, height=600)
    fig.show()

plot_roc_curves(results)

# Precision-Recall curves (plotly)
def plot_pr_curves(results):
    fig = go.Figure()
    for model_name, result in results.items():
        try:
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ap = auc(recall, precision)
            fig.add_trace(go.Scatter(x=recall, y=precision, name=f'{model_name} (AUC={ap:.4f})', mode='lines'))
        except Exception:
            pass
    fig.update_layout(title='Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision', width=900, height=600)
    fig.show()

plot_pr_curves(results)

# Feature importances top-10 for models that provide them
fi_list = []
for name, res in results.items():
    fi = res.get('feature_importance', None)
    if fi is not None:
        fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False).head(10)
        df_fi = pd.DataFrame({'feature': fi_series.index, 'importance': fi_series.values})
        df_fi['model'] = name
        fi_list.append(df_fi)
if fi_list:
    df_fi_all = pd.concat(fi_list, ignore_index=True)
    display(df_fi_all.sort_values(['model','importance'], ascending=[True, False]).reset_index(drop=True))
else:
    print("No feature importances available for displayed models.")

# ---------------- Accuracy summary (predicted vs actual) ----------------
def accuracy_summary(results, y_test, save_csv=True, outname='accuracy_summary.csv'):
    print("\n" + "="*60)
    print(" ACCURACY REPORT (Predicted vs Actual)")
    print("="*60)

    acc_list = []
    for model_name, res in results.items():
        y_pred = res['predictions']
        acc = accuracy_score(y_test, y_pred)
        acc_list.append({"Model": model_name, "Accuracy": round(acc, 4), "Accuracy (%)": round(acc * 100, 2)})
        print(f"{model_name}: {acc:.4f} ({acc*100:.2f}%)")

    df_acc = pd.DataFrame(acc_list).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    print("\nFinal Accuracy Table:")
    display(df_acc)

    # Plot bar chart
    plt.figure(figsize=(8,4))
    sns.barplot(data=df_acc, x='Accuracy', y='Model')
    plt.title('Model Accuracy (Predicted vs Actual)')
    plt.xlim(0,1.0)
    plt.xlabel('Accuracy')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()       

    if save_csv:
        df_acc.to_csv(outname, index=False)
        print(f"Saved accuracy CSV -> {outname}")

    return df_acc

df_accuracy = accuracy_summary(results, y_test)

print("\nDone. Files produced: 'model_comparison_report.csv', 'accuracy_summary.csv' (in current working directory).")
