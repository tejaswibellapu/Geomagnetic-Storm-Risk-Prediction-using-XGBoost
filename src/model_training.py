import os
import json
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "model_training.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTraining")

def train_risk_model(features_data_path, models_dir, outputs_dir):
    """Trains an XGBoost Regressor to predict Geomagnetic Risk Score 6 hours ahead using lag and rolling features."""
    logger.info("Starting XGBoost 6-hour ahead risk model training...")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    if not os.path.exists(features_data_path):
        raise FileNotFoundError(f"Features data not found: {features_data_path}")
        
    df = pd.read_csv(features_data_path)
    logger.info(f"Loaded feature dataset with {len(df)} records.")
    
    # 1. Define Target Variable: Geomagnetic Risk Score 6 hours ahead (0 to 10)
    df['target_Kp_6h_ahead'] = df['Kp'].shift(-6)
    df = df.dropna(subset=['target_Kp_6h_ahead'])
    df['geomagnetic_risk_score'] = df['target_Kp_6h_ahead'] * (10.0 / 9.0)
    df['geomagnetic_risk_score'] = df['geomagnetic_risk_score'].clip(0.0, 10.0)
    
    # 2. Select Features (historical and rolling geomagnetic variables)
    feature_cols = [
        'Kp', 'Dst', 'ap',
        'Kp_rolling_mean_6h', 'Kp_rolling_std_6h',
        'Dst_rolling_mean_6h', 'Dst_rolling_std_6h',
        'Kp_lag_6h', 'Dst_lag_6h', 'ap_lag_6h',
        'Kp_trend_6h_vs_24h', 'Dst_trend_6h_vs_24h'
    ]
    
    logger.info(f"Selected features for model: {feature_cols}")
    
    X = df[feature_cols]
    y = df['geomagnetic_risk_score']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 4. Cross Validation and Hyperparameter Tuning
    logger.info("Initializing Grid Search CV for XGBoost Regressor...")
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0]
    }
    
    # Using 5-fold cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    logger.info(f"Best hyperparameters found: {best_params}")
    
    best_model = grid_search.best_estimator_
    
    # 5. Evaluate Model
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Also calculate classification accuracy mapping to 4 categories
    def get_risk_cat(score):
        if score < 3.0:
            return 0
        elif score < 6.0:
            return 1
        elif score < 9.0:
            return 2
        else:
            return 3
    
    y_test_cat = np.array([get_risk_cat(val) for val in y_test])
    y_pred_cat = np.array([get_risk_cat(val) for val in y_pred])
    class_acc = float(np.mean(y_test_cat == y_pred_cat))
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'classification_accuracy': class_acc,
        'best_params': best_params
    }
    
    # Save metrics report
    metrics_path = os.path.join(outputs_dir, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Model evaluation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, ClassAcc={class_acc * 100:.2f}%")
    
    # Save the trained model
    model_path = os.path.join(models_dir, "xgboost_risk_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved trained XGBoost model to: {model_path}")
    
    # 6. Plot and Save Feature Importance
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_cols)[sorted_idx], importance[sorted_idx], color='teal')
    plt.xlabel("XGBoost Feature Importance")
    plt.title("Geomagnetic 6-Hour Ahead Risk Prediction Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "feature_importance.png"), dpi=150)
    plt.close()
    logger.info("Saved feature importance chart.")

def train_tec_model(features_data_path, models_dir, outputs_dir):
    """Trains an XGBoost Regressor to predict Total Electron Content (TEC) 3 hours ahead using lag and rolling features."""
    logger.info("Starting XGBoost 3-hour ahead TEC model training...")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    if not os.path.exists(features_data_path):
        raise FileNotFoundError(f"Features data not found: {features_data_path}")
        
    df = pd.read_csv(features_data_path)
    logger.info(f"Loaded feature dataset with {len(df)} records.")
    
    # 1. Define Target Variable: TEC 3 hours ahead
    df['target_TEC_3h_ahead'] = df['TEC'].shift(-3)
    df = df.dropna(subset=['target_TEC_3h_ahead', 'TEC_lag_1h', 'TEC_lag_2h', 'TEC_lag_3h'])
    
    # 2. Select Features (historical and rolling space weather and ionospheric variables)
    feature_cols = [
        'Kp', 'Dst', 'ap', 'TEC',
        'TEC_lag_1h', 'TEC_lag_2h', 'TEC_lag_3h',
        'Kp_lag_1h', 'Kp_lag_2h', 'Kp_lag_3h',
        'Dst_lag_1h', 'Dst_lag_2h', 'Dst_lag_3h',
        'ap_lag_1h', 'ap_lag_2h', 'ap_lag_3h',
        'Kp_rolling_mean_6h', 'Dst_rolling_mean_6h', 'TEC_rolling_mean_6h',
        'Kp_rolling_std_6h', 'Dst_rolling_std_6h', 'TEC_rolling_std_6h',
        'Kp_lag_6h', 'Dst_lag_6h', 'ap_lag_6h', 'TEC_lag_6h',
        'Kp_trend_6h_vs_24h', 'Dst_trend_6h_vs_24h', 'TEC_trend_6h_vs_24h'
    ]
    
    logger.info(f"Selected features for TEC model: {feature_cols}")
    
    X = df[feature_cols]
    y = df['target_TEC_3h_ahead']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 4. Cross Validation and Hyperparameter Tuning
    logger.info("Initializing Grid Search CV for XGBoost Regressor (TEC)...")
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0]
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    logger.info(f"Best hyperparameters found for TEC: {best_params}")
    best_model = grid_search.best_estimator_
    
    # 5. Evaluate Model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage of predictions within +/- 5 TECU
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5.0) * 100
    tec_range = y_test.max() - y_test.min()
    accuracy_range = (1 - (mae / tec_range)) * 100
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'PercentageWithin5TECU': float(within_5),
        'RangeBasedAccuracy': float(accuracy_range),
        'best_params': best_params
    }
    
    # Save metrics report
    metrics_path = os.path.join(outputs_dir, "tec_model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"TEC Model evaluation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Within5TECU={within_5:.2f}%, RangeAcc={accuracy_range:.2f}%")
    
    # Save the trained model
    model_path = os.path.join(models_dir, "xgboost_tec_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved trained XGBoost TEC model to: {model_path}")
    
    # 6. Plot and Save Feature Importance
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_cols)[sorted_idx], importance[sorted_idx], color='purple')
    plt.xlabel("XGBoost Feature Importance")
    plt.title("TEC 3-Hour Ahead Prediction Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "tec_feature_importance.png"), dpi=150)
    plt.close()
    logger.info("Saved TEC feature importance chart.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    features_data_path = os.path.join(data_dir, "space_weather_features.csv")
    train_risk_model(features_data_path, models_dir, outputs_dir)
    train_tec_model(features_data_path, models_dir, outputs_dir)

if __name__ == "__main__":
    main()
