import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Swing+ Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #1f77b4;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.swing-plus-score {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
}
.interpretation {
    font-size: 1.2rem;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train_model():
    """Load data and train the LightGBM model"""
    try:
        # Load data
        data = pd.read_csv("swing+ data.csv")
        
        # Define features and target
        features = ['avg_swing_speed', 'avg_swing_length', 'attack_angle', 
                   'attack_direction', 'vertical_swing_path']
        target = 'xwobacon'
        
        X = data[features]
        y = data[target]
        weights = data['pa']
        
        # Split data with weights
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        
        # Train LightGBM model
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.1,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
        valid_data = lgb.Dataset(X_test, label=y_test, weight=weights_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Calculate normalization parameters for swing+
        predictions = model.predict(X)
        weighted_mean = np.average(predictions, weights=data['pa'])
        weighted_variance = np.average((predictions - weighted_mean)**2, weights=data['pa'])
        weighted_std = np.sqrt(weighted_variance)
        
        return model, weighted_mean, weighted_std, data, features
        
    except FileNotFoundError:
        st.error("Could not find 'swing+ data.csv'. Please make sure the data file is in the same directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def predict_swing_plus(model, weighted_mean, weighted_std, input_data):
    """Make prediction and convert to swing+ score"""
    # Make prediction
    prediction = model.predict([input_data])[0]
    
    # Convert to z-score
    z_score = (prediction - weighted_mean) / weighted_std
    
    # Transform to swing+ (mean=100, std=10)
    swing_plus = z_score * 10 + 100
    
    return prediction, swing_plus

def get_swing_plus_interpretation(swing_plus_score):
    """Return interpretation of swing+ score"""
    if swing_plus_score >= 120:
        return "🔥 Elite", "#ff4b4b"
    elif swing_plus_score >= 110:
        return "⭐ Above Average", "#ff8c42"
    elif swing_plus_score >= 90:
        return "✓ Average", "#00cc88"
    elif swing_plus_score >= 80:
        return "⚠️ Below Average", "#ffaa00"
    else:
        return "❌ Poor", "#ff4b4b"

def create_gauge_chart(swing_plus_score):
    """Create a gauge chart for swing+ score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = swing_plus_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Swing+ Score"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 150]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "lightgray"},
                {'range': [80, 90], 'color': "yellow"},
                {'range': [90, 110], 'color': "lightgreen"},
                {'range': [110, 120], 'color': "orange"},
                {'range': [120, 150], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 120
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">⚾ Swing+ Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, weighted_mean, weighted_std, data, features = load_and_train_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.header("🎯 Enter Swing Metrics")
    st.sidebar.markdown("Adjust the sliders to input your swing characteristics:")
    
    # Get data ranges for sliders
    if data is not None:
        ranges = {
            'avg_swing_speed': (data['avg_swing_speed'].min(), data['avg_swing_speed'].max()),
            'avg_swing_length': (data['avg_swing_length'].min(), data['avg_swing_length'].max()),
            'attack_angle': (data['attack_angle'].min(), data['attack_angle'].max()),
            'attack_direction': (data['attack_direction'].min(), data['attack_direction'].max()),
            'vertical_swing_path': (data['vertical_swing_path'].min(), data['vertical_swing_path'].max())
        }
        
        defaults = {
            'avg_swing_speed': data['avg_swing_speed'].median(),
            'avg_swing_length': data['avg_swing_length'].median(),
            'attack_angle': data['attack_angle'].median(),
            'attack_direction': data['attack_direction'].median(),
            'vertical_swing_path': data['vertical_swing_path'].median()
        }
    else:
        # Fallback ranges if data not available
        ranges = {
            'avg_swing_speed': (60.0, 85.0),
            'avg_swing_length': (6.0, 9.0),
            'attack_angle': (-20.0, 40.0),
            'attack_direction': (-30.0, 30.0),
            'vertical_swing_path': (10.0, 25.0)
        }
        defaults = {
            'avg_swing_speed': 72.5,
            'avg_swing_length': 7.5,
            'attack_angle': 10.0,
            'attack_direction': 0.0,
            'vertical_swing_path': 17.5
        }
    
    # Input fields with validation
    avg_swing_speed = st.sidebar.number_input(
        "Average Swing Speed (mph)",
        min_value=float(ranges['avg_swing_speed'][0]),
        max_value=float(ranges['avg_swing_speed'][1]),
        value=float(defaults['avg_swing_speed']),
        step=0.1,
        format="%.1f",
        help="The average speed of your swing"
    )
    
    avg_swing_length = st.sidebar.number_input(
        "Average Swing Length (ft)",
        min_value=float(ranges['avg_swing_length'][0]),
        max_value=float(ranges['avg_swing_length'][1]),
        value=float(defaults['avg_swing_length']),
        step=0.1,
        format="%.1f",
        help="The average length/distance of your swing path"
    )
    
    attack_angle = st.sidebar.number_input(
        "Attack Angle (degrees)",
        min_value=float(ranges['attack_angle'][0]),
        max_value=float(ranges['attack_angle'][1]),
        value=float(defaults['attack_angle']),
        step=0.1,
        format="%.1f",
        help="The angle of attack at contact (positive = uppercut)"
    )
    
    attack_direction = st.sidebar.number_input(
        "Attack Direction (degrees)",
        min_value=float(ranges['attack_direction'][0]),
        max_value=float(ranges['attack_direction'][1]),
        value=float(defaults['attack_direction']),
        step=0.1,
        format="%.1f",
        help="The horizontal direction of attack"
    )
    
    vertical_swing_path = st.sidebar.number_input(
        "Vertical Swing Path (degrees)",
        min_value=float(ranges['vertical_swing_path'][0]),
        max_value=float(ranges['vertical_swing_path'][1]),
        value=float(defaults['vertical_swing_path']),
        step=0.1,
        format="%.1f",
        help="The vertical component of your swing path"
    )
    
    # Prediction button
    if st.sidebar.button("🎯 Calculate Swing+ Score", type="primary"):
        input_data = [avg_swing_speed, avg_swing_length, attack_angle, attack_direction, vertical_swing_path]
        
        # Make prediction
        predicted_xwobacon, swing_plus_score = predict_swing_plus(model, weighted_mean, weighted_std, input_data)
        
        # Store results in session state
        st.session_state.predicted_xwobacon = predicted_xwobacon
        st.session_state.swing_plus_score = swing_plus_score
        st.session_state.input_data = input_data
    
    # Display results if available
    if hasattr(st.session_state, 'swing_plus_score'):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Prediction Results")
            
            # Swing+ score with interpretation
            interpretation, color = get_swing_plus_interpretation(st.session_state.swing_plus_score)
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="swing-plus-score" style="color: {color};">
                    {st.session_state.swing_plus_score:.1f}
                </div>
                <div class="interpretation" style="color: {color};">
                    {interpretation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            st.metric(
                label="Predicted xwOBACON", 
                value=f"{st.session_state.predicted_xwobacon:.3f}",
                help="Expected weighted On-Base Average on Contact"
            )
        
        with col2:
            st.markdown("### 📊 Swing+ Gauge")
            gauge_fig = create_gauge_chart(st.session_state.swing_plus_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Score interpretation guide
        st.markdown("### 📋 Swing+ Score Guide")
        interpretation_df = pd.DataFrame({
            'Score Range': ['120+', '110-119', '90-109', '80-89', '<80'],
            'Rating': ['Elite', 'Above Average', 'Average', 'Below Average', 'Poor'],
            'Percentile': ['Top 2.5%', 'Top ~16%', 'Middle 68%', 'Bottom ~16%', 'Bottom 2.5%'],
            'Description': [
                'Outstanding swing mechanics',
                'Better than most players',
                'League average performance',
                'Room for improvement',
                'Significant development needed'
            ]
        })
        
        st.dataframe(interpretation_df, use_container_width=True, hide_index=True)
        
        # Input summary
        st.markdown("### 📝 Your Input Summary")
        input_summary_df = pd.DataFrame({
            'Metric': ['Average Swing Speed', 'Average Swing Length', 'Attack Angle', 'Attack Direction', 'Vertical Swing Path'],
            'Value': [f"{val:.1f}" for val in st.session_state.input_data],
            'Unit': ['mph', 'ft', 'degrees', 'degrees', 'degrees']
        })
        st.dataframe(input_summary_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("👈 Enter your swing metrics in the sidebar and click 'Calculate Swing+ Score' to see your prediction!")
        
        # Show example data ranges
        if data is not None:
            st.markdown("### 📊 Data Ranges Reference")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Swing Speed", f"{data['avg_swing_speed'].mean():.1f} mph", f"Range: {data['avg_swing_speed'].min():.1f}-{data['avg_swing_speed'].max():.1f}")
                st.metric("Avg Swing Length", f"{data['avg_swing_length'].mean():.1f} ft", f"Range: {data['avg_swing_length'].min():.1f}-{data['avg_swing_length'].max():.1f}")
            
            with col2:
                st.metric("Attack Angle", f"{data['attack_angle'].mean():.1f}°", f"Range: {data['attack_angle'].min():.1f}-{data['attack_angle'].max():.1f}")
                st.metric("Attack Direction", f"{data['attack_direction'].mean():.1f}°", f"Range: {data['attack_direction'].min():.1f}-{data['attack_direction'].max():.1f}")
            
            with col3:
                st.metric("Vertical Swing Path", f"{data['vertical_swing_path'].mean():.1f}°", f"Range: {data['vertical_swing_path'].min():.1f}-{data['vertical_swing_path'].max():.1f}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Swing+ is a normalized metric with a mean of 100 and standard deviation of 10, where higher scores indicate better predicted performance.*")

if __name__ == "__main__":
    main()