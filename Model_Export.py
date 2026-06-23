import pandas as pd
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import warnings

# Ignore XGBoost version warnings during conversion
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# =========================
# CONFIG
# =========================
TRAIN_FILES = {4: "./trainset_window_4.csv", 12: "./trainset_window_12.csv", 20: "./trainset_window_20.csv"}
TEST_FILES = {4: "./testset_window_4.csv", 12: "./testset_window_12.csv", 20: "./testset_window_20.csv"}

LABELS = ["P0000", "P0562", "P0113", "P0102", "P0403", "P0404", "P2562", "P2015", "P2009", "P0107", "P0069", "P0089", "P0234", "P0406"]

# Custom thresholds for noisy labels
THRESHOLDS = {
    "DEFAULT": 0.3
}

RANDOM_STATE = 42

# =========================
# DATA UTILITIES
# =========================
def process_dataframe(path):
    df = pd.read_csv(path)
    y_list = []
    for l in LABELS:
        # Convert label column to binary (1 if code exists, 0 otherwise)
        col_bin = (df[l].fillna(0).values > 0).astype(int) if l in df.columns else np.zeros(len(df), dtype=int)
        y_list.append(col_bin)
    y = np.column_stack(y_list)
    
    # Define features to ignore (Targets and metadata)
    ignore = LABELS 
    X = df.drop(columns=[c for c in ignore if c in df.columns], errors='ignore')
    X = X.loc[:, ~X.columns.str.contains("^Unnamed")]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y

def train_model(X_train, y_train_label):
    # Handle class imbalance automatically
    pos_count = np.sum(y_train_label == 1)
    neg_count = np.sum(y_train_label == 0)
    ratio = (neg_count / pos_count) if pos_count > 0 else 1
    
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=ratio,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train_label)
    return model

# =========================
# MAIN EXPORT PIPELINE
# =========================
def main():
    for window in [4, 12, 20]:
        print(f"\n" + "="*70)
        print(f"🌀 PROCESSING WINDOW SIZE: {window}")
        print("="*70)
        
        try:
            X_train, y_train = process_dataframe(TRAIN_FILES[window])
            X_test, y_test = process_dataframe(TEST_FILES[window])
        except FileNotFoundError as e:
            print(f"❌ Skipping window {window}: {e}")
            continue

        # Ensure test set has same columns as train set
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # 1. SAVE FEATURE ORDER (Required for Flutter data alignment)
        feature_filename = f"feature_order_w{window}.txt"
        with open(feature_filename, "w") as f:
            f.write(",".join(X_train.columns.tolist()))
        print(f"📄 Saved feature list: {feature_filename}")

        preds_list = []
        feature_names_onnx = [f'f{k}' for k in range(X_train.shape[1])]

        for i, label in enumerate(LABELS):
            y_tr_col = y_train[:, i]
            
            # Skip training if only one class is present in the training set
            if len(np.unique(y_tr_col)) < 2:
                print(f"⚠️ Skipping {label}: Only one class present.")
                preds_list.append(np.full(len(X_test), y_tr_col[0]))
                continue

            print(f"🚀 Training & Exporting Specialist: {label}")
            model = train_model(X_train, y_tr_col)
            
            # --- THE FIX FOR ONNX & EVALUATION ---
            # Rename booster features to generic f0, f1... to satisfy ONNX conversion
            model.get_booster().feature_names = feature_names_onnx

            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
            try:
                onnx_model = onnxmltools.convert_xgboost(
                    model, 
                    initial_types=initial_type,
                    target_opset=13
                )
                onnx_filename = f"model_{label}_w{window}.onnx"
                onnxmltools.utils.save_model(onnx_model, onnx_filename)
                print(f"✅ Exported: {onnx_filename}")
            except Exception as e:
                print(f"❌ ONNX Export failed for {label}: {e}")

            # Prediction for metrics (Must use renamed columns to match the updated booster)
            X_test_renamed = X_test.copy()
            X_test_renamed.columns = feature_names_onnx
            ts_proba = model.predict_proba(X_test_renamed)[:, 1]
            
            # Apply thresholding
            thresh = THRESHOLDS.get(label, THRESHOLDS["DEFAULT"])
            preds_list.append((ts_proba > thresh).astype(int))

        # --- LOGICAL OVERRIDE & EVALUATION ---
        y_test_pred = np.column_stack(preds_list)

        # If no faults detected, default to "P0000" (Normal)
        no_pred_mask = np.sum(y_test_pred, axis=1) == 0
        y_test_pred[no_pred_mask, 0] = 1 

        print(f"\n{'P-Code':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 60)
        
        f1_list = []
        for i, label in enumerate(LABELS):
            y_true, y_pred = y_test[:, i], y_test_pred[:, i]
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_list.append(f1)
            print(f"{label:<8} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {int(np.sum(y_true)):<8}")

        subset_acc = np.mean(np.all(y_test == y_test_pred, axis=1))
        print("-" * 60)
        print(f"📊 Mean F1: {np.mean(f1_list):.4f}  |  🎯 Exact Match: {subset_acc:.4f}")

    print("\n🎉 DONE! All specialists exported for mobile deployment.")

if __name__ == "__main__":
    main()
