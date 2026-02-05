# 101.py

import argparse
import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from groq import Groq
import json
import re
import platform
import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import models module (could be in current directory or parent)
try:
    from models import run_classification_models, run_regression_models, save_best_model
except ImportError:
    # Try importing from parent directory
    sys.path.append('..')
    from models import run_classification_models, run_regression_models, save_best_model

# 1. CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True)
parser.add_argument('--target', required=True)
args = parser.parse_args()

# 2. Set Groq API key
# Try to find api_key.txt in current directory or parent directory
api_key_path = None
if os.path.exists('api_key.txt'):
    api_key_path = 'api_key.txt'
elif os.path.exists('../api_key.txt'):
    api_key_path = '../api_key.txt'
elif os.path.exists('../../api_key.txt'):
    api_key_path = '../../api_key.txt'
else:
    # If no api_key.txt found, check for environment variable
    if 'GROQ_API_KEY' not in os.environ:
        raise FileNotFoundError("api_key.txt not found and GROQ_API_KEY environment variable not set")

if api_key_path:
    with open(api_key_path) as f:
        os.environ['GROQ_API_KEY'] = f.read().strip()

# 3. Load data
df = pd.read_csv(args.csv)
y = df[args.target]
X = df.drop(columns=[args.target])

# 1.5. Detect task type
if y.dtype == 'object' or y.nunique() < 20:
    task = 'classification'
else:
    task = 'regression'

def basic_stats(X):
    num = X.select_dtypes(include=np.number)
    cat = X.select_dtypes(exclude=np.number)
    stats = {
        'numerical': {
            col: {
                'mean': float(num[col].mean()),
                'std': float(num[col].std()),
                'min': float(num[col].min()),
                'max': float(num[col].max())
            }
            for col in num.columns
        },
        'categorical': {
            col: cat[col].value_counts().head(3).to_dict()
            for col in cat.columns
        }
    }
    return stats

def try_parse_json(content):
    try:
        return json.loads(content)
    except Exception:
        # Try to fix common issues: remove trailing commas, etc.
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        try:
            return json.loads(content)
        except Exception:
            return None

def call_llama_for_features(stats, min_features=5):
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    prompt = (
        f"You are a data scientist. Given the following summary of a dataset's features, "
        f"suggest at least {min_features} new feature engineering ideas for a machine learning task. "
        f"Respond ONLY with a valid JSON object, where each key is a new feature name and each value is a short description. "
        f"Do not include any explanation or text outside the JSON object.\n\n"
        f"Example:\n"
        f'{{"Fare_Log": "Log of Fare column", "Family_Size": "Sum of SibSp and Parch columns", "Is_Alone": "1 if Family_Size==0 else 0", "Title": "Extracted from Name column", "Cabin_Initial": "First letter of Cabin"}}\n\n'
        f"Feature summary:\n{json.dumps(stats, indent=2)}"
    )
    for attempt in range(2):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful data scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        suggestions = try_parse_json(content)
        if isinstance(suggestions, dict) and len(suggestions) >= min_features:
            return suggestions
        else:
            print("[LLM] Not enough features suggested or JSON error, retrying...")
            print("[LLM RAW OUTPUT]", content)
            prompt = "Your last response was not valid JSON. Please output only a valid JSON object as specified."
    print("[LLM] Failed to get valid feature suggestions.")
    return {}

def call_llama_for_code(var_name, description, X_columns, custom_prompt=None):
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = (
            f"Write a single line of Python pandas code to create a new feature '{var_name}' in DataFrame X. "
            f"The feature is: {description}. Use only these columns: {X_columns}. "
            f"Assume X is a pandas DataFrame. Only output the code line, nothing else."
        )
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=100
    )
    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return code

def execute_code(code, X):
    local_vars = {'X': X, 'pd': pd, 'np': np}
    try:
        if "get_dummies" in code:
            # Extract the column to encode
            match = re.search(r"get_dummies\(X\['(\w+)'\]", code)
            if match:
                col = match.group(1)
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
                return X
        exec(code, {}, local_vars)
        return local_vars['X']
    except Exception as e:
        raise e

def do_rfe(X, y, task, n_features=7):
    if task == 'classification':
        estimator = RandomForestClassifier()
    else:
        estimator = RandomForestRegressor()
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_]

def encode_categoricals(X):
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = X_encoded[col].astype('category').cat.codes
    return X_encoded

# Main loop
all_new_features = []
feature_creation_log = []  # <-- Add this to log feature creation attempts
model_results_log = []     # <-- Add this to log model results
best_score = 0
best_model = None
best_model_name = None

max_loops = 5
loop_count = 0
no_improve_count = 0
last_best_score = 0

while True:
    stats = basic_stats(X)
    feature_ideas = call_llama_for_features(stats)
    for var_name, desc in feature_ideas.items():
        if var_name in X.columns or var_name in all_new_features:
            continue
        code = call_llama_for_code(var_name, desc, list(X.columns))
        try:
            X = execute_code(code, X)
            all_new_features.append(var_name)
            feature_creation_log.append({
                "feature": var_name,
                "description": desc,
                "code": code,
                "status": "success"
            })
        except Exception as e:
            # Retry with error message
            error_msg = str(e)
            retry_prompt = (
                f"Your last code for feature '{var_name}' failed with error: {error_msg}. "
                f"Please output a corrected single line of code to create this feature in DataFrame X. "
                f"Feature description: {desc}. Use only these columns: {list(X.columns)}. "
                f"Assume X is a pandas DataFrame. Only output the code line, nothing else."
            )
            retry_code = call_llama_for_code(var_name, desc, list(X.columns), custom_prompt=retry_prompt)
            try:
                X = execute_code(retry_code, X)
                all_new_features.append(var_name)
                feature_creation_log.append({
                    "feature": var_name,
                    "description": desc,
                    "code": retry_code,
                    "status": "success (retry)"
                })
            except Exception as e2:
                feature_creation_log.append({
                    "feature": var_name,
                    "description": desc,
                    "code": retry_code,
                    "status": f"failed after retry: {e2}"
                })

    # 6. RFE
    X_encoded = encode_categoricals(X)
    X_encoded = X_encoded.replace([np.inf, -np.inf], 0)  # <--- Move this up, before RFE
    X_encoded = X_encoded.fillna(0)                      # <--- And fill NaNs here too
    selected = do_rfe(X_encoded, y, task)
    X_selected = X_encoded[selected]

    # 7. Train/test models
    if task == 'classification':
        results = run_classification_models(X_selected, y)
    else:
        results = run_regression_models(X_selected, y)

    # 8. Select best model
    for name, res in results.items():
        model_results_log.append({
            "model": name,
            "score": res['score'],
            "cv_scores": res['cv_scores']
        })
        if res['score'] > best_score:
            best_score = res['score']
            best_model = res['model']
            best_model_name = name

    print(f"Best model: {best_model_name}, Score: {best_score}")

    loop_count += 1

    # Early stopping if no improvement
    if best_score > last_best_score:
        no_improve_count = 0
        last_best_score = best_score
    else:
        no_improve_count += 1

    if best_score >= 0.85:
        break
    elif loop_count >= max_loops:
        print(f"Reached max loop count ({max_loops}). Stopping.")
        break
    elif no_improve_count >= 3:
        print("No improvement in score for 3 consecutive iterations. Stopping.")
        break
    else:
        print("Score < 0.85, engineering more features...")

# 9. Save best model
save_best_model(best_model, 'best_model.pkl')
print("Best model saved as best_model.pkl")

# Data overview
data_overview = {
    "num_rows": len(df),
    "num_columns": len(df.columns),
    "missing_values": df.isnull().sum().to_dict(),
    "class_balance": y.value_counts(normalize=True).to_dict() if task == "classification" else None,
    "feature_types": {
        "numerical": list(df.select_dtypes(include=np.number).columns),
        "categorical": list(df.select_dtypes(exclude=np.number).columns)
    }
}

# Run metadata
run_metadata = {
    "agenticml_version": "0.1",
    "run_time": datetime.datetime.now().isoformat(),
    "python_version": platform.python_version(),
    "os": platform.platform()
}

# Best model params
best_model_params = getattr(best_model, "get_params", lambda: {})()

# Final report
report = {
    "run_metadata": run_metadata,
    "data_overview": data_overview,
    "task": task,
    "input_csv": args.csv,
    "target_column": args.target,
    "engineered_features": feature_creation_log,
    "selected_features": list(selected),
    "model_results": model_results_log,
    "best_model": {
        "name": best_model_name,
        "score": best_score,
        "params": best_model_params
    }
}
with open("report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Report saved as report.json")
