import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures

# -------------------------
# Load and Train Model Once
# -------------------------

@st.cache_resource
def train_model():
    data = pd.read_csv("Training Data.csv")

    base_features = [
        'avg_intercept_y_vs_batter', 'avg_bat_speed', 'avg_swing_length',
        'attack_angle', 'attack_direction', 'swing_tilt'
    ]
    target = 'xwobacon'
    weights = data['competitive_swings']

    X_base = data[base_features]
    y = data[target]

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    feature_names = poly.get_feature_names_out(base_features).tolist()

    dtrain = xgb.DMatrix(X_poly, label=y, weight=weights, feature_names=feature_names)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    model = xgb.train(params, dtrain, num_boost_round=200)

    # Save mean/std for Swing+
    mean_pred = y.mean()
    std_pred = y.std()

    return model, poly, base_features, feature_names, mean_pred, std_pred


model, poly, base_features, feature_names, mean_pred, std_pred = train_model()

# -------------------------
# Prediction Function
# -------------------------

def predict_swing_plus(hitter_stats: dict):
    X_input = np.array([[hitter_stats[f] for f in base_features]])
    X_poly = poly.transform(X_input)
    dmatrix = xgb.DMatrix(X_poly, feature_names=feature_names)
    pred = model.predict(dmatrix)[0]
    swing_plus = (pred - mean_pred) / std_pred * 10 + 100
    return pred, swing_plus

# -------------------------
# Streamlit UI
# -------------------------

st.title("Swing+ Predictor ⚾")
st.markdown("Input a hitter’s swing metrics to get their **predicted xwOBAcon** and **Swing+ score**.")

inputs = {}
for f in base_features:
    inputs[f] = st.number_input(f"Enter {f}", value=0.0)

if st.button("Predict Swing+"):
    pred, swing_plus = predict_swing_plus(inputs)
    st.success(f"Predicted xwOBAcon: **{pred:.3f}**")
    st.success(f"Swing+ Score: **{swing_plus:.0f}**")
