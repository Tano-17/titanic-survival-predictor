"""
🏆 Weekly Project: Titanic Survival Predictor
=============================================
AI Engineering Mentorship — Week 3

Deliverable: Python script + evaluation report (printed to terminal)

Pipeline:
  1. Load and clean data (handle missing Age, Cabin, Embarked)
  2. Engineer at least 2 new features (FamilySize, IsAlone, Title)
  3. Train 3 models: Logistic Regression, Decision Tree, Random Forest
  4. Compare using accuracy, precision, recall, and F1
  5. Print a final report showing which model won and why

Stretch goals:
  - Tune Random Forest hyperparameters using GridSearchCV
  - Plot a learning curve to visualise train vs test performance gap
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD AND CLEAN THE DATA
# ============================================================
def load_data():
    """Load the Titanic train and test datasets."""
    print("=" * 60)
    print("📂 STEP 1: Loading Data")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'test.csv'))

    print(f"  Training set shape: {train_df.shape}")
    print(f"  Test set shape:     {test_df.shape}")
    print(f"\n  Missing values in training set:")
    missing = train_df.isnull().sum()
    print(missing[missing > 0].to_string().replace('\n', '\n  '))
    print()

    return train_df, test_df


def clean_data(df):
    """
    Handle missing values:
      - Age: fill with median (grouped by Pclass and Sex for accuracy)
      - Cabin: too many missing — drop the column or create a 'HasCabin' flag
      - Embarked: fill with mode (most common port)
    """
    print("🧹 STEP 2: Cleaning Data")
    print("-" * 40)

    # --- Age: fill with median grouped by Pclass & Sex ---
    age_before = df['Age'].isnull().sum()
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback: if any still missing, use overall median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    print(f"  Age: filled {age_before} missing values (grouped median by Pclass & Sex)")

    # --- Cabin: create HasCabin flag, then drop Cabin ---
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    cabin_missing = df['Cabin'].isnull().sum()
    df.drop('Cabin', axis=1, inplace=True)
    print(f"  Cabin: {cabin_missing} missing → created 'HasCabin' flag, dropped column")

    # --- Embarked: fill with mode ---
    embarked_before = df['Embarked'].isnull().sum()
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    print(f"  Embarked: filled {embarked_before} missing values (mode = '{df['Embarked'].mode()[0]}')")

    # --- Fare: fill if missing (mostly in test set) ---
    if df['Fare'].isnull().sum() > 0:
        fare_missing = df['Fare'].isnull().sum()
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        print(f"  Fare: filled {fare_missing} missing values (median)")

    print(f"  ✅ Remaining missing values: {df.isnull().sum().sum()}\n")
    return df


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """
    Create new features:
      1. FamilySize = SibSp + Parch + 1
      2. IsAlone = 1 if FamilySize == 1
      3. Title extracted from Name
    """
    print("⚙️  STEP 3: Feature Engineering")
    print("-" * 40)

    # --- Feature 1: FamilySize ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print(f"  ✅ FamilySize = SibSp + Parch + 1  (range: {df['FamilySize'].min()}-{df['FamilySize'].max()})")

    # --- Feature 2: IsAlone ---
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print(f"  ✅ IsAlone  (alone passengers: {df['IsAlone'].sum()} / {len(df)})")

    # --- Feature 3: Title from Name ---
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Group rare titles
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don',
         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace(['Ms', 'Mme'], 'Mrs')

    print(f"  ✅ Title extracted and grouped:")
    print(f"     {df['Title'].value_counts().to_dict()}")

    print()
    return df


def encode_features(df):
    """Encode categorical variables for model training."""
    print("🔢 STEP 4: Encoding Categorical Variables")
    print("-" * 40)

    # Map Sex to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    print("  ✅ Sex → 0/1 (male/female)")

    # One-hot encode Embarked and Title
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], dtype=int)
    print("  ✅ One-hot encoded: Embarked, Title")

    # Drop columns that won't help the model
    drop_cols = ['PassengerId', 'Name', 'Ticket']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print(f"  ✅ Dropped: {drop_cols}")

    print(f"  Final feature set: {list(df.columns)}\n")
    return df


# ============================================================
# STEP 3: TRAIN MODELS
# ============================================================
def prepare_data(df):
    """Split data into features (X) and target (y), then train/test split."""
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"📊 Data split: {X_train.shape[0]} train / {X_test.shape[0]} test (80/20, stratified)\n")
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train Logistic Regression, Decision Tree, and Random Forest."""
    print("=" * 60)
    print("🤖 STEP 5: Training Models")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  Training: {name}...")

        # Use scaled data for Logistic Regression, raw for tree-based
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1 Score:  {f1:.4f}")

    return results


# ============================================================
# STEP 4: COMPARE & REPORT
# ============================================================
def print_final_report(results, y_test):
    """Print a comprehensive comparison report."""
    print("\n")
    print("=" * 60)
    print("📋 FINAL EVALUATION REPORT")
    print("=" * 60)

    # Build comparison table
    print("\n┌─────────────────────┬──────────┬───────────┬──────────┬──────────┐")
    print("│ Model               │ Accuracy │ Precision │ Recall   │ F1 Score │")
    print("├─────────────────────┼──────────┼───────────┼──────────┼──────────┤")

    for name, metrics in results.items():
        print(f"│ {name:<19} │ {metrics['accuracy']:.4f}   │ {metrics['precision']:.4f}    │ {metrics['recall']:.4f}   │ {metrics['f1']:.4f}   │")

    print("└─────────────────────┴──────────┴───────────┴──────────┴──────────┘")

    # Determine winner
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_model_name]

    print(f"\n🏆 WINNER: {best_model_name}")
    print(f"   F1 Score: {best['f1']:.4f} | Accuracy: {best['accuracy']:.4f}")
    print()

    # Explain why
    print("📝 WHY THIS MODEL WON:")
    print("-" * 40)

    if best_model_name == 'Random Forest':
        print("   Random Forest won because it combines multiple decision trees")
        print("   (ensemble method) which reduces overfitting and captures complex")
        print("   non-linear relationships in the data. It handles mixed feature")
        print("   types well and is robust to outliers.")

    elif best_model_name == 'Logistic Regression':
        print("   Logistic Regression won because, despite being a simpler model,")
        print("   it performs well when features have been properly scaled and the")
        print("   decision boundary is approximately linear. It generalises better")
        print("   than more complex models on this dataset size.")

    elif best_model_name == 'Decision Tree':
        print("   Decision Tree won by finding clear decision boundaries in the data.")
        print("   It handles categorical features naturally and captures non-linear")
        print("   patterns. However, watch for overfitting on larger datasets.")

    # Show detailed classification report for the winner
    print(f"\n📊 Detailed Classification Report ({best_model_name}):")
    print("-" * 40)
    print(classification_report(y_test, best['y_pred'], target_names=['Did Not Survive', 'Survived']))

    return best_model_name


# ============================================================
# STRETCH GOAL 1: GridSearchCV for Random Forest
# ============================================================
def tune_random_forest(X_train, y_train, X_test, y_test):
    """Tune Random Forest hyperparameters using GridSearchCV."""
    print("\n" + "=" * 60)
    print("🔧 STRETCH GOAL: Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1',
        n_jobs=1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV F1 score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred_tuned = grid_search.best_estimator_.predict(X_test)
    tuned_f1 = f1_score(y_test, y_pred_tuned)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    print(f"  Test F1 score:    {tuned_f1:.4f}")
    print(f"  Test Accuracy:    {tuned_acc:.4f}")

    return grid_search.best_estimator_


# ============================================================
# STRETCH GOAL 2: Learning Curve
# ============================================================
def plot_learning_curve(model, X, y, title="Learning Curve"):
    """Plot a learning curve to visualise train vs test performance gap."""
    print(f"\n📈 Plotting learning curve for {title}...")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2196F3')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='#FF5722')
    plt.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training F1')
    plt.plot(train_sizes, test_mean, 'o-', color='#FF5722', label='Validation F1')

    plt.title(f'Learning Curve — {title}', fontsize=14, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, 'learning_curve.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {save_path}\n")


# ============================================================
# STRETCH GOAL 3: Kaggle Submission
# ============================================================
def generate_submission(model, train_cols, test_df):
    print("\n" + "=" * 60)
    print("🚀 STRETCH GOAL 3: Kaggle Submission")
    print("=" * 60)
    print("  Processing test.csv for prediction...\n")
    
    passenger_ids = test_df['PassengerId'].copy()
    
    df = clean_data(test_df.copy())
    df = engineer_features(df)
    df = encode_features(df)
    
    # Align columns
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
            
    df = df[train_cols]
    predictions = model.predict(df)
    
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sub_path = os.path.join(base_dir, 'submission.csv')
    submission.to_csv(sub_path, index=False)
    print(f"\n  ✅ Kaggle predictions saved successfully to:\n     {sub_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "🚢" * 30)
    print("   TITANIC SURVIVAL PREDICTOR")
    print("🚢" * 30 + "\n")

    # 1. Load data
    train_df, test_df = load_data()

    # 2. Clean data
    train_df = clean_data(train_df)

    # 3. Feature engineering
    train_df = engineer_features(train_df)

    # 4. Encode categorical features
    train_df = encode_features(train_df)
    
    train_cols = [c for c in train_df.columns if c != 'Survived']

    # 5. Prepare train/test split
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(train_df)

    # 6. Train models
    results = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

    # 7. Print final report
    winner = print_final_report(results, y_test)

    # 8. Stretch: Tune Random Forest
    tuned_rf = tune_random_forest(X_train, y_train, X_test, y_test)

    # 9. Stretch: Plot learning curve
    plot_learning_curve(tuned_rf, X_train, y_train, title="Tuned Random Forest")
    
    # 10. Stretch: Generate Kaggle submission
    generate_submission(tuned_rf, train_cols, test_df)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
