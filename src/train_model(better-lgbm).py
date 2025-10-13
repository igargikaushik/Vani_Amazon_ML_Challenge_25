
import sys

import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import QuantileTransformer
from lightgbm import early_stopping

# Import your feature extraction modules
from text_features import engineer_text_features
from image_features import extract_comprehensive_image_features
from sklearn.model_selection import train_test_split

# ---for Dynamic path handling ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(SCRIPT_DIR, '..', 'dataset')
if not os.path.exists(DATASET_FOLDER):
    #fallback for Kaggle env
    DATASET_FOLDER = '/kaggle/input/smartproductpricing/EcomProductPricing/dataset'
OUTPUT_PATH = '/kaggle/working/test_out.csv'
# --- Configuration ---
#DATASET_FOLDER = 'dataset'
#TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv') # Assuming you have this file
#TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')   # Assuming you have this file
#OUTPUT_PATH = os.path.join(DATASET_FOLDER, 'test_out.csv')

# --- SMAPE Metric Function ---
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    # Clip predictions to prevent division by zero or log(0) issues
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def train_and_predict_pipeline():
    # --- 1. Data Loading ---
    print("🚀 Loading Training and Test Data...")
    try:
        # Load training and test data
        train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
        
        # Verify price column exists in training data
        if 'price' not in train_df.columns:
            print("❌ ERROR: 'price' column not found in training data.")
            print("   Please ensure train.csv contains the 'price' column.")
            return

    except FileNotFoundError as e:
        print(f"❌ Error: Required data file not found: {e}")
        return

    # --- 2. Target Preprocessing ---
    # We predict the log of (1 + price) for a better distribution
    Y_train_log = np.log1p(train_df['price'])
    
    # --- 3. Feature Engineering ---
    print("\n📝 Extracting Text Features...")
    # Fit text features on training data
    X_train_text_sparse, _, tfidf_vectorizer, _ = engineer_text_features(
        train_df, 
        fit_tfidf=True, 
        analyze_importance=False
    )
    # Transform test data using fitted transformers
    X_test_text_sparse, _, _, _ = engineer_text_features(
        test_df, 
        fit_tfidf=False, 
        tfidf_vectorizer=tfidf_vectorizer,
        analyze_importance=False
    )
    #align the test features to train
    X_test_text_sparse = X_test_text_sparse[:, :X_train_text_sparse.shape[1]]



    # --- 4. Image Feature Extraction ---
    print("\n🖼️ Extracting Image Features...")
    X_train_image, _ = extract_comprehensive_image_features(
        train_df, use_deep_features=True, model_name='efficientnet'
    )
    X_test_image, _ = extract_comprehensive_image_features(
        test_df, use_deep_features=True
    )
    #align the test img features to train's 
    X_test_image = X_test_image[:, :X_train_image.shape[1]]
    # --- Save extracted image features for future reuse ---
    np.save('X_train_image.npy', X_train_image)
    np.save('X_test_image.npy', X_test_image)
    print("✅ Saved image features: X_train_image.npy, X_test_image.npy")


    # Cast to float64 for consistency
    X_train_image = X_train_image.astype(np.float32)
    X_test_image = X_test_image.astype(np.float32)

    # --- 5. Feature Integration (Combining) ---
    print("\n🔗 Combining Text and Image Features...")


    # Combine sparse text features with dense image features
    X_train = hstack([X_train_text_sparse, X_train_image])
    X_test = hstack([X_test_text_sparse, X_test_image])
    #  Ensure train and test feature alignment

    if X_test.shape[1] != X_train.shape[1]:
        print(f"⚠️ Feature mismatch detected! Aligning test features ({X_test.shape[1]}) → train ({X_train.shape[1]})")
        from scipy.sparse import csr_matrix
        if X_test.shape[1] < X_train.shape[1]:
          diff = X_train.shape[1] - X_test.shape[1]
          X_test = hstack([X_test, csr_matrix((X_test.shape[0], diff))])
        elif X_test.shape[1] > X_train.shape[1]:
          X_test = X_test[:, :X_train.shape[1]]
        print(f"✅ After alignment: train={X_train.shape}, test={X_test.shape}")
    
    assert X_train.shape[1] == X_test.shape[1], (
        f"❌ Feature count mismatch: Train={X_train.shape[1]}, Test={X_test.shape[1]}"
    )
    print(f"Final Train Feature Shape: {X_train.shape}")
    print(f"Final Test Feature Shape: {X_test.shape}")
    print("\n💾 Saving final combined features...")
    np.save('train_features.npy', X_train.toarray())
    np.save('test_features.npy', X_test.toarray())
    print("✅ Saved: train_features.npy, test_features.npy")


    # --- 6. Model Training (LightGBM) ---
    print("\n Training LightGBM Regressor...")
    lgbm = lgb.LGBMRegressor(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        objective='regression_l1', 
        metric='mae',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.4,
        min_child_samples=30,
        n_jobs=-1,
        random_state=42
    )



    lgbm.fit(
      X_train, Y_train_log,
      eval_set=[(X_train, Y_train_log)],
      eval_metric='mae',
      callbacks=[early_stopping(stopping_rounds=50)]
)

    # --- 7. Prediction ---
    print("\n🔮 Generating Predictions...")
    
    # Predict on the log scale
    Y_pred_log = lgbm.predict(X_test)
    
    # Inverse transform: Exponentiate to get back to the original price scale
    Y_pred = np.expm1(Y_pred_log)
    
    # Clip predictions to ensure non-negativity
    Y_pred = np.clip(Y_pred, a_min=0, a_max=np.percentile(Y_pred, 99.5))

    # --- 8. Evaluation (on training set for sanity check) ---
    Y_train_pred_log = lgbm.predict(X_train)
    Y_train_pred = np.expm1(Y_train_pred_log)
    
    smape_score = smape(train_df['price'], Y_train_pred)
    print(f"\n✅ Training Set SMAPE Score: {smape_score:.4f}%")
    
    # --- 9. Submission File Generation ---
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else np.arange(len(Y_pred)),

        'price': Y_pred
    })
    
    # Save predictions
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n Predictions saved to {OUTPUT_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Price range: ${Y_pred.min():.2f} - ${Y_pred.max():.2f}")
    print(f"Sample predictions:\n{submission_df.head()}")
    
    print("\n Training and prediction pipeline completed successfully!")
    print(" Ready for submission to ML Challenge!")


if __name__ == "__main__":
    train_and_predict_pipeline()
  
