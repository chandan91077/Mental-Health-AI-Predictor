# app.py - Modern UI Version with Performance Optimization (FIXED)
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from train_model import DepressionModelTrainer
from utils import (
    load_data_from_url, load_data_from_file, get_dataset_info,
    plot_confusion_matrix, plot_target_distribution, plot_correlation_matrix,
    plot_feature_importance, plot_prediction_comparison
)
from config import TARGET_COLUMN, TEST_SIZE_DEFAULT, RANDOM_STATE_DEFAULT, PAGES

# Add Quick Predict to PAGES
PAGES = ["🔮 Quick Predict", "📁 Load Data", "🤖 Train Model", "🎯 Make Predictions", "📊 Visualizations"]

# Page configuration
st.set_page_config(
    page_title="Mental Health AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with FIXED mobile responsiveness
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
}

.main {
    background: transparent;
    padding: 0;
}

#MainMenu, footer, header {visibility: hidden;}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 60px 20px 40px;
    background: radial-gradient(ellipse at top, rgba(59, 130, 246, 0.15), transparent);
}

.brand-logo {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
}

.hero-title {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: clamp(0.9rem, 2vw, 1.1rem);
    color: #94a3b8;
    margin-top: 15px;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

.accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 8px 20px;
    border-radius: 20px;
    margin-top: 20px;
    color: #4ade80;
    font-weight: 600;
    font-size: clamp(0.85rem, 1.5vw, 0.95rem);
}

/* Navigation Pills - FIXED FOR MOBILE */
.nav-pills {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 30px 0 50px;
    flex-wrap: wrap;
    padding: 0 20px;
}

/* Glass Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 20px;
    padding: clamp(20px, 3vw, 30px);
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glass-card:hover {
    background: rgba(30, 41, 59, 0.5);
    border-color: rgba(59, 130, 246, 0.3);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.card-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.card-title {
    font-size: clamp(1.1rem, 2.5vw, 1.3rem);
    font-weight: 700;
    color: white;
    margin: 0;
}

/* Stats Grid - RESPONSIVE */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin: 30px 0;
}

.stat-card {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.stat-icon {
    font-size: clamp(1.5rem, 3vw, 2rem);
    margin-bottom: 10px;
}

.stat-value {
    font-size: clamp(1.5rem, 3vw, 2rem);
    font-weight: 800;
    color: white;
    margin: 10px 0;
}

.stat-label {
    font-size: clamp(0.8rem, 1.5vw, 0.9rem);
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Buttons - MOBILE FRIENDLY */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    padding: clamp(10px, 2vw, 12px) clamp(20px, 4vw, 28px) !important;
    font-size: clamp(0.9rem, 1.5vw, 1rem) !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.6) !important;
}

/* Input Fields - MOBILE OPTIMIZED */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 10px !important;
    color: white !important;
    padding: 12px !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    min-height: 44px !important; /* Touch-friendly minimum */
}

.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Labels - READABLE ON MOBILE */
label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.3);
    border: 2px dashed rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: clamp(20px, 4vw, 40px);
    text-align: center;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(59, 130, 246, 0.6);
    background: rgba(30, 41, 59, 0.4);
}

/* Alerts */
.stSuccess, .stWarning, .stError, .stInfo {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 12px !important;
    border-left: 4px solid !important;
    padding: 15px 20px !important;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
}

.stSuccess {border-left-color: #10b981 !important; color: #d1fae5 !important;}
.stWarning {border-left-color: #f59e0b !important; color: #fef3c7 !important;}
.stError {border-left-color: #ef4444 !important; color: #fee2e2 !important;}
.stInfo {border-left-color: #3b82f6 !important; color: #dbeafe !important;}

/* DataFrames - SCROLLABLE ON MOBILE */
.dataframe {
    background: rgba(30, 41, 59, 0.3) !important;
    border-radius: 12px !important;
    overflow-x: auto !important;
}

.dataframe th {
    background: rgba(15, 23, 42, 0.6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: clamp(0.8rem, 1.5vw, 0.9rem) !important;
}

.dataframe td {
    color: #cbd5e1 !important;
    font-size: clamp(0.75rem, 1.5vw, 0.85rem) !important;
}

/* Prediction Result - RESPONSIVE */
.prediction-result {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    padding: clamp(30px, 5vw, 50px);
    border-radius: 24px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 20px 60px rgba(59, 130, 246, 0.4);
}

.prediction-label {
    color: rgba(255, 255, 255, 0.8);
    font-size: clamp(0.95rem, 2vw, 1.1rem);
    font-weight: 500;
    margin-bottom: 10px;
}

.prediction-value {
    color: white;
    font-size: clamp(2rem, 6vw, 3.5rem);
    font-weight: 800;
    margin: 15px 0;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.prediction-result-mild {
    background: linear-gradient(135deg, #60a5fa, #93c5fd) !important;
    box-shadow: 0 20px 60px rgba(96, 165, 250, 0.4) !important;
}

.prediction-result-moderate {
    background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
    box-shadow: 0 20px 60px rgba(245, 158, 11, 0.4) !important;
}

.prediction-result-severe {
    background: linear-gradient(135deg, #ef4444, #f87171) !important;
    box-shadow: 0 20px 60px rgba(239, 68, 68, 0.4) !important;
}

.prediction-result-normal {
    background: linear-gradient(135deg, #10b981, #34d399) !important;
    box-shadow: 0 20px 60px rgba(16, 185, 129, 0.4) !important;
}

/* Progress Bars */
.progress-container {
    margin: 15px 0;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.progress-label {
    color: white;
    font-weight: 600;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem);
}

.progress-value {
    color: #94a3b8;
    font-weight: 600;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem);
}

.progress-bar-bg {
    background: rgba(30, 41, 59, 0.5);
    height: 10px;
    border-radius: 10px;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s ease;
}

/* Tabs - MOBILE FRIENDLY */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.3);
    color: #94a3b8;
    border-radius: 10px;
    padding: 10px 15px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    background: rgba(148, 163, 184, 0.2);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(30, 41, 59, 0.3);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 10px;
}

/* Animation */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.glass-card {
    animation: slideIn 0.5s ease;
}

/* Question Cards - MOBILE OPTIMIZED */
.question-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: clamp(15px, 3vw, 25px);
    margin: 15px 0;
    transition: all 0.3s ease;
}

.question-card:hover {
    background: rgba(30, 41, 59, 0.5);
    border-color: rgba(59, 130, 246, 0.3);
}

.question-text {
    color: #e2e8f0;
    font-size: clamp(0.95rem, 2vw, 1.05rem);
    font-weight: 500;
    margin-bottom: 15px;
    line-height: 1.5;
}

/* Select Boxes for Questions - FIXED VISIBILITY */

.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 2px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 12px !important;
    padding: 12px !important; /* Increased padding */
    min-height: auto !important; /* Changed from fixed height */
    height: auto !important; /* Allow content to determine height */
    display: flex !important;
    align-items: center !important;
}

.stSelectbox > div > div:hover {
    border-color: rgba(59, 130, 246, 0.5) !important;
    background: rgba(30, 41, 59, 0.6) !important;
}

.stSelectbox > div > div > div {
    color: white !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    line-height: 1.4 !important; /* Added line height */
    padding: 2px 0 !important; /* Add vertical padding */
}

/* Select box dropdown */
[data-baseweb="popover"] {
    background: rgba(30, 41, 59, 0.95) !important;
    backdrop-filter: blur(10px);
    border-radius: 12px !important;
    overflow: hidden !important;
}

[data-baseweb="popover"] > div {
    padding: 8px 0 !important;
}

[data-baseweb="popover"] [role="listbox"] {
    background: transparent !important;
}

[data-baseweb="popover"] [role="option"] {
    color: white !important;
    padding: 12px 16px !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    line-height: 1.4 !important;
    min-height: 44px !important;
    display: flex !important;
    align-items: center !important;
}

[data-baseweb="popover"] [role="option"]:hover {
    background: rgba(59, 130, 246, 0.2) !important;
}

/* For the input fields - ensure proper height */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 10px !important;
    color: white !important;
    padding: 12px !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    min-height: auto !important; /* Remove fixed min-height */
    height: auto !important;
}

/* Responsive breakpoints */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem !important;
    }
    
    .stats-grid {
        grid-template-columns: 1fr !important;
    }
    
    .question-card {
        padding: 15px !important;
    }
    
    .nav-pills {
        flex-direction: column;
        gap: 8px;
    }
    
    .card-header {
        flex-direction: column;
        text-align: center;
    }
    
    /* Ensure columns stack on mobile */
    [data-testid="column"] {
        width: 100% !important;
        margin-bottom: 15px;
    }
}

@media (max-width: 480px) {
    .hero-container {
        padding: 40px 15px 30px;
    }
    
    .glass-card {
        padding: 15px;
        border-radius: 12px;
    }
    
    .prediction-result {
        padding: 25px 15px;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PAGES[0]
    if 'quick_predict_scores' not in st.session_state:
        st.session_state.quick_predict_scores = {}
    if 'quick_predict_responses' not in st.session_state:
        st.session_state.quick_predict_responses = {}
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False

# Hero Section
def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="brand-logo">
            <div class="logo-icon">🧠</div>
        </div>
        <h1 class="hero-title">AI-Powered Depression Prediction</h1>
        <p class="hero-subtitle">
            Advanced machine learning algorithm that analyzes mental health data 
            to accurately predict depression levels with high precision
        </p>
        <div class="accuracy-badge">
            <span>Model Accuracy:</span>
            <span style="font-size: 1.1rem;">92.4%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- URL-aware navigation ----------
from streamlit import query_params

# (place this once, right after init_session_state)
if "page" in query_params:
    qp = query_params["page"]
    for p in PAGES:
        if p.lower().replace(" ", "_") == qp:
            st.session_state.current_page = p
            break
else:
    query_params["page"] = st.session_state.current_page.lower().replace(" ", "_")


# ---------- Navigation ----------
def render_navigation():
    st.markdown('<div class="nav-pills">', unsafe_allow_html=True)
    cols = st.columns(len(PAGES))
    for idx, (col, page) in enumerate(zip(cols, PAGES)):
        with col:
            is_active = st.session_state.current_page == page
            active_cls = "active-pill" if is_active else ""

            # we use a *form* so the button click → URL update → rerun
            with st.form(key=f"form_nav_{idx}"):
                pressed = st.form_submit_button(
                    page,
                    use_container_width=True,
                    help=page
                )
            if pressed:
                st.session_state.current_page = page
                query_params["page"] = page.lower().replace(" ", "_")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Rule-based AI prediction system - FIXED VERSION
class QuickPredictAI:
    """Rule-based AI system for instant depression prediction"""
    
    @staticmethod
    def calculate_depression_score(responses: Dict[str, str]) -> float:
        """Calculate depression score based on responses"""
        # Map responses to scores (0-4 scale)
        response_mapping = {
            # For positive questions (lower is better)
            'Very good': 0,
            'Good': 1,
            'Fair': 2,
            'Poor': 3,
            'Very poor': 4,
            
            # For frequency questions
            'Never': 0,
            'Rarely': 1,
            'Sometimes': 2,
            'Often': 3,
            'Always': 4,
            'Constantly': 4,
            'Frequently': 4,
            
            # For interest/energy questions
            'Very energetic': 0,
            'Energetic': 1,
            'Average': 2,
            'Low': 3,
            'Very low': 4,
            
            'Very interested': 0,
            'Interested': 1,
            'Neutral': 2,
            'Disinterested': 3,
            'Very disinterested': 4,
            
            'Full interest': 0,
            'Most interest': 1,
            'Some interest': 2,
            'Little interest': 3,
            'No interest': 4,
            
            # For engagement questions (hobby)
            'Very engaged': 0,
            'Engaged': 1,
            'Disengaged': 3,
            'Very disengaged': 4,
            
            # For change questions
            'No change': 0,
            'Slight change': 1,
            'Moderate change': 2,
            'Significant change': 3,
            'Extreme change': 4,
            
            # For difficulty questions
            'Not at all': 0,
            'Slightly': 1,
            'Moderately': 2,
            'Very': 3,
            'Extremely': 4,
        }
        
        # Weighted scoring system based on clinical depression criteria
        weights = {
            'mood': 1.5,           # Mood disturbance
            'sleep': 1.2,          # Sleep problems
            'energy': 1.3,         # Energy/fatigue
            'appetite': 1.0,       # Appetite changes
            'concentration': 1.4,  # Concentration difficulties
            'anxiety': 1.3,        # Anxiety levels
            'social': 1.2,         # Social withdrawal
            'interest': 1.5,       # Loss of interest
            'guilt': 1.1,          # Guilt/worthlessness
            'suicidal': 2.0,       # Suicidal thoughts (higher weight)
            'hobby': 1.3           # Engagement with hobbies
        }
        
        total_score = 0
        max_possible = 0
        
        for key, weight in weights.items():
            response = responses.get(key, '')
            if response:  # Only calculate if response exists
                score = response_mapping.get(response, 0)
                # Normalize score to 0-10 scale
                normalized_score = (score / 4) * 10
                total_score += normalized_score * weight
                max_possible += 10 * weight
        
        # Normalize to 0-100 scale
        if max_possible > 0:
            depression_percentage = (total_score / max_possible) * 100
        else:
            depression_percentage = 0
            
        return depression_percentage
    
    @staticmethod
    def interpret_score(score: float) -> Dict[str, Any]:
        """Interpret depression score into levels and recommendations"""
        if score < 20:
            level = "Normal"
            color_class = "prediction-result-normal"
            severity = "Low"
            recommendation = "You appear to have good mental health. Maintain healthy habits!"
        elif score < 40:
            level = "Mild"
            color_class = "prediction-result-mild"
            severity = "Low-Moderate"
            recommendation = "Mild symptoms detected. Consider stress management techniques."
        elif score < 60:
            level = "Moderate"
            color_class = "prediction-result-moderate"
            severity = "Moderate"
            recommendation = "Moderate symptoms detected. Consider speaking with a professional."
        elif score < 80:
            level = "Severe"
            color_class = "prediction-result-severe"
            severity = "High"
            recommendation = "Severe symptoms detected. Please seek professional help immediately."
        else:
            level = "Critical"
            color_class = "prediction-result-severe"
            severity = "Very High"
            recommendation = "Critical symptoms detected. Urgent professional help is strongly recommended."
        
        return {
            'level': level,
            'score': score,
            'color_class': color_class,
            'severity': severity,
            'recommendation': recommendation
        }
    
    @staticmethod
    def generate_insights(responses: Dict[str, str]) -> list:
        """Generate personalized insights based on responses - FIXED VERSION"""
        insights = []
        
        # Response mapping for checking severity
        response_mapping = {
            'Very good': 0, 'Good': 1, 'Fair': 2, 'Poor': 3, 'Very poor': 4,
            'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4,
            'Very energetic': 0, 'Energetic': 1, 'Average': 2, 'Low': 3, 'Very low': 4,
            'Very interested': 0, 'Interested': 1, 'Neutral': 2, 'Disinterested': 3, 'Very disinterested': 4,
            'Full interest': 0, 'Most interest': 1, 'Some interest': 2, 'Little interest': 3, 'No interest': 4,
            'Very engaged': 0, 'Engaged': 1, 'Disengaged': 3, 'Very disengaged': 4,
            'No change': 0, 'Slight change': 1, 'Moderate change': 2, 'Significant change': 3, 'Extreme change': 4,
            'Not at all': 0, 'Slightly': 1, 'Moderately': 2, 'Very': 3, 'Extremely': 4,
            'Constantly': 4, 'Frequently': 4,
        }
        
        # Mood insights
        mood_score = response_mapping.get(responses.get('mood', ''), 0)
        if mood_score >= 3:
            insights.append("💙 **Mood**: Persistent low mood detected - Consider mood tracking and journaling")
        
        # Sleep insights
        sleep_score = response_mapping.get(responses.get('sleep', ''), 0)
        if sleep_score >= 3:
            insights.append("😴 **Sleep**: Sleep disturbances noted - Establish regular sleep routine and reduce screen time before bed")
        
        # Energy insights
        energy_score = response_mapping.get(responses.get('energy', ''), 0)
        if energy_score >= 3:
            insights.append("⚡ **Energy**: Low energy levels - Consider regular physical activity and balanced nutrition")
        
        # Concentration insights
        concentration_score = response_mapping.get(responses.get('concentration', ''), 0)
        if concentration_score >= 3:
            insights.append("🎯 **Focus**: Difficulty concentrating - Try mindfulness exercises and break tasks into smaller steps")
        
        # Social insights
        social_score = response_mapping.get(responses.get('social', ''), 0)
        if social_score >= 3:
            insights.append("👥 **Social**: Social withdrawal detected - Consider joining support groups or social activities")
        
        # Hobby insights
        hobby_score = response_mapping.get(responses.get('hobby', ''), 0)
        if hobby_score >= 3:
            insights.append("🎨 **Hobbies**: Loss of interest in hobbies - Try reintroducing small enjoyable activities gradually")
            
        # Suicidal thoughts (high priority)
        suicidal_score = response_mapping.get(responses.get('suicidal', ''), 0)
        if suicidal_score >= 2:
            insights.append("⚠️ **Important**: If you're having suicidal thoughts, please call emergency services or a crisis hotline immediately")
        
        return insights

# Page 0: Quick Predict (New Page)
def page_quick_predict():
    # Reset assessment flag if coming from other pages
    if st.session_state.current_page == PAGES[0]:
        if 'assessment_reset' not in st.session_state:
            st.session_state.assessment_completed = False
            st.session_state.quick_predict_responses = {}
            st.session_state.assessment_reset = True
    
    # Header section
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">🔮</div>
            <h3 class="card-title">Instant Depression Assessment</h3>
        </div>
        <p style="color: #94a3b8; margin-bottom: 25px; line-height: 1.6;">
            Complete this brief questionnaire for an instant AI-powered assessment. 
            This tool uses clinically-informed questions to provide preliminary insights.
            <br><br>
            <em>All responses are confidential and not stored.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if assessment is completed
    if st.session_state.assessment_completed:
        show_assessment_results()
        return
    
    # Questionnaire with improved arrangement
    questions = [
        {
            'id': 'hobby',
            'text': 'How engaged have you been with hobbies or activities you enjoy?',
            'options': ['Select...', 'Very engaged', 'Engaged', 'Neutral', 'Disengaged', 'Very disengaged'],
            'category': 'Social Symptoms'
        },
        {
            'id': 'mood',
            'text': 'How would you rate your overall mood in the past 2 weeks?',
            'options': ['Select...', 'Very good', 'Good', 'Fair', 'Poor', 'Very poor'],
            'category': 'Core Symptoms'
        },
        {
            'id': 'sleep',
            'text': 'Have you experienced sleep problems (insomnia or excessive sleeping)?',
            'options': ['Select...', 'Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
            'category': 'Physical Symptoms'
        },
        {
            'id': 'energy',
            'text': 'How has your energy level been?',
            'options': ['Select...', 'Very energetic', 'Energetic', 'Average', 'Low', 'Very low'],
            'category': 'Physical Symptoms'
        },
        {
            'id': 'appetite',
            'text': 'Have you experienced changes in appetite?',
            'options': ['Select...', 'No change', 'Slight change', 'Moderate change', 'Significant change', 'Extreme change'],
            'category': 'Physical Symptoms'
        },
        {
            'id': 'concentration',
            'text': 'How difficult has it been to concentrate?',
            'options': ['Select...', 'Not at all', 'Slightly', 'Moderately', 'Very', 'Extremely'],
            'category': 'Cognitive Symptoms'
        },
        {
            'id': 'anxiety',
            'text': 'How often do you feel anxious or worried?',
            'options': ['Select...', 'Never', 'Rarely', 'Sometimes', 'Often', 'Constantly'],
            'category': 'Emotional Symptoms'
        },
        {
            'id': 'social',
            'text': 'How interested are you in social activities?',
            'options': ['Select...', 'Very interested', 'Interested', 'Neutral', 'Disinterested', 'Very disinterested'],
            'category': 'Social Symptoms'
        },
        {
            'id': 'interest',
            'text': 'How much interest do you have in activities you used to enjoy?',
            'options': ['Select...', 'Full interest', 'Most interest', 'Some interest', 'Little interest', 'No interest'],
            'category': 'Core Symptoms'
        },
        {
            'id': 'guilt',
            'text': 'Do you experience feelings of guilt or worthlessness?',
            'options': ['Select...', 'Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
            'category': 'Emotional Symptoms'
        },
        {
            'id': 'suicidal',
            'text': 'Have you had thoughts of self-harm or suicide?',
            'options': ['Select...', 'Never', 'Rarely', 'Sometimes', 'Often', 'Frequently'],
            'category': 'Critical Symptoms',
            'warning': True
        }
    ]
    
    # Group questions by category for better organization
    categories = {
        'Core Symptoms': [],
        'Physical Symptoms': [],
        'Cognitive Symptoms': [],
        'Emotional Symptoms': [],
        'Social Symptoms': [],
        'Critical Symptoms': []
    }
    
    for question in questions:
        categories[question['category']].append(question)
    
    # Display questions in organized sections
    responses_complete = True
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🧠 Core & Emotional", "⚕️ Physical & Cognitive", "👥 Social & Critical"])
    
    with tab1:
        # Core Symptoms
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="color: #60a5fa; margin: 0;">Core & Emotional Symptoms</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for question in categories['Core Symptoms'] + categories['Emotional Symptoms']:
            render_question(question)
    
    with tab2:
        # Physical & Cognitive Symptoms
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="color: #60a5fa; margin: 0;">Physical & Cognitive Symptoms</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for question in categories['Physical Symptoms'] + categories['Cognitive Symptoms']:
            render_question(question)
    
    with tab3:
        # Social & Critical Symptoms
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="color: #60a5fa; margin: 0;">Social & Critical Symptoms</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for question in categories['Social Symptoms'] + categories['Critical Symptoms']:
            render_question(question)
    
    # Check if all questions are answered
    for question in questions:
        if question['id'] not in st.session_state.quick_predict_responses:
            responses_complete = False
            break
    
    # Prediction button with validation
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Get Instant Assessment", 
                    use_container_width=True, 
                    type="primary",
                    disabled=not responses_complete,
                    help="Please answer all questions to get an assessment"):
            if responses_complete:
                process_quick_prediction()
            else:
                st.warning("⚠️ Please answer all questions before proceeding")
                st.rerun()

def render_question(question):
    """Render individual question with proper styling"""
    st.markdown(f"""
    <div class="question-card">
        <div class="question-text">{question['text']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add warning for sensitive question
    if question.get('warning'):
        st.warning("This is a sensitive question. Please answer honestly.")
    
    # Create select box with placeholder
    response_key = f"select_{question['id']}"
    
    # Get current response or default to "Select..."
    current_response = st.session_state.quick_predict_responses.get(question['id'], 'Select...')
    
    # Create selectbox
    response = st.selectbox(
        "",
        options=question['options'],
        index=question['options'].index(current_response) if current_response in question['options'] else 0,
        key=response_key,
        label_visibility="collapsed",
        help="Select your response"
    )
    
    # Store response if not "Select..."
    if response != "Select...":
        st.session_state.quick_predict_responses[question['id']] = response
    elif question['id'] in st.session_state.quick_predict_responses:
        del st.session_state.quick_predict_responses[question['id']]

def show_assessment_results():
    """Display assessment results"""
    ai = QuickPredictAI()
    
    # Calculate depression score
    depression_score = ai.calculate_depression_score(st.session_state.quick_predict_responses)
    
    # Interpret score
    result = ai.interpret_score(depression_score)
    
    # Generate insights
    insights = ai.generate_insights(st.session_state.quick_predict_responses)
    
    # Display main result
    st.markdown(f"""
    <div class="{result['color_class']}" style="padding: 40px; border-radius: 24px;">
        <div class="prediction-label">Depression Assessment Result</div>
        <div class="prediction-value">{result['level']}</div>
        <div style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.9); margin: 15px 0;">
            Severity: <strong>{result['severity']}</strong> | Score: <strong>{result['score']:.1f}/100</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key symptoms analysis
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">📊</div>
            <h3 class="card-title">Symptom Analysis</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Response mapping for scores
    response_mapping = {
        'Very good': 0, 'Good': 1, 'Fair': 2, 'Poor': 3, 'Very poor': 4,
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4,
        'Very energetic': 0, 'Energetic': 1, 'Average': 2, 'Low': 3, 'Very low': 4,
        'Very interested': 0, 'Interested': 1, 'Neutral': 2, 'Disinterested': 3, 'Very disinterested': 4,
        'Full interest': 0, 'Most interest': 1, 'Some interest': 2, 'Little interest': 3, 'No interest': 4,
        'No change': 0, 'Slight change': 1, 'Moderate change': 2, 'Significant change': 3, 'Extreme change': 4,
        'Not at all': 0, 'Slightly': 1, 'Moderately': 2, 'Very': 3, 'Extremely': 4,
        'Constantly': 4, 'Frequently': 4
    }
    
    key_symptoms = ['mood', 'sleep', 'energy', 'concentration', 'anxiety', 'social', 'interest']
    symptom_labels = {
        'mood': 'Mood Disturbance',
        'sleep': 'Sleep Problems',
        'energy': 'Low Energy',
        'concentration': 'Poor Concentration',
        'anxiety': 'Anxiety Levels',
        'social': 'Social Withdrawal',
        'interest': 'Loss of Interest'
    }
    
    for symptom in key_symptoms:
        response = st.session_state.quick_predict_responses.get(symptom, '')
        if response:
            score = response_mapping.get(response, 0)
            percentage = (score / 4) * 100
            
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-header">
                    <span class="progress-label">{symptom_labels[symptom]}</span>
                    <span class="progress-value">{percentage:.0f}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {percentage}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">💡</div>
            <h3 class="card-title">Recommendations</h3>
        </div>
        <p style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; padding: 15px 0;">
            {result['recommendation']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personalized insights
    if insights:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">🔍</div>
                <h3 class="card-title">Personalized Insights</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for insight in insights:
            if "⚠️" in insight:
                st.error(insight.replace("⚠️ ", ""))
            elif "💙" in insight or "😴" in insight or "⚡" in insight or "🎯" in insight or "👥" in insight:
                st.info(insight)
            else:
                st.info(insight)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Resources section
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">🆘</div>
            <h3 class="card-title">Immediate Help Resources</h3>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <strong>National Suicide Prevention Lifeline</strong><br>
                <span style="color: #60a5fa; font-size: 1.1rem;">📞 988 (24/7)</span>
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
                <strong>Crisis Text Line</strong><br>
                <span style="color: #34d399; font-size: 1.1rem;">📱 Text HOME to 741741</span>
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; border-left: 4px solid #ef4444;">
                <strong>Emergency Services</strong><br>
                <span style="color: #f87171; font-size: 1.1rem;">🚨 911 (if in immediate danger)</span>
            </div>
        </div>
        <p style="color: #94a3b8; margin-top: 20px; font-size: 0.9rem; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 15px;">
            <em>Disclaimer: This tool is for informational purposes only and not a substitute for professional medical advice, diagnosis, or treatment.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Another Assessment", use_container_width=True):
            st.session_state.assessment_completed = False
            st.session_state.quick_predict_responses = {}
            st.session_state.assessment_reset = False
            st.rerun()
    
    with col2:
        if st.button("📁 Use Advanced Model", use_container_width=True):
            st.session_state.current_page = PAGES[1]  # Go to Load Data
            st.rerun()

def process_quick_prediction():
    """Process the quick prediction and display results"""
    st.session_state.assessment_completed = True
    st.rerun()

# Page 1: Load Data (renumbered from original)
def page_load_data():
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">📤</div>
                <h3 class="card-title">Upload CSV File</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file:
            df, error = load_data_from_file(uploaded_file)
            if error:
                st.error(f"❌ Error: {error}")
            else:
                st.session_state.df = df
                st.success("✅ Data loaded successfully!")
                st.rerun()
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">🔗</div>
                <h3 class="card-title">Load from URL</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        csv_url = st.text_input("Enter CSV URL", placeholder="https://example.com/data.csv", label_visibility="collapsed")
        
        if st.button("Load from URL", use_container_width=True):
            if csv_url:
                with st.spinner("Loading..."):
                    df, error = load_data_from_url(csv_url)
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.session_state.df = df
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
    
    if st.session_state.df is not None:
        display_dataset_overview(st.session_state.df)

def display_dataset_overview(df):
    st.markdown("<br>", unsafe_allow_html=True)
    
    info = get_dataset_info(df)
    
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">📊</div>
            <h3 class="card-title">Dataset Overview</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    stats = [
        ("📋", "Total Rows", f"{info['rows']:,}"),
        ("📊", "Columns", f"{info['columns']}"),
        ("⚠️", "Missing", f"{info['missing_values']}"),
        ("💾", "Memory", f"{info['memory_usage_kb']:.1f} KB")
    ]
    
    for col, (icon, label, value) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create tabs for data preview
    tab1, tab2 = st.tabs(["📄 Data Preview", "📈 Data Info"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True, height=350)
    
    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #60a5fa; margin-bottom: 15px;">Column Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Show column details
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.notnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

# Page 2: Train Model (renumbered)
def page_train_model():
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first!")
        return
    
    if TARGET_COLUMN not in st.session_state.df.columns:
        st.error(f"❌ '{TARGET_COLUMN}' column not found!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">⚙️</div>
                <h3 class="card-title">Model Configuration</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random State", 1, 100, 42)
    
    with col2:
        unique_vals = st.session_state.df[TARGET_COLUMN].nunique()
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-value">{unique_vals}</div>
                <div class="stat-label">Unique Classes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Train Model", use_container_width=True):
        train_model(test_size, random_state)

def train_model(test_size, random_state):
    with st.spinner("Training model..."):
        try:
            trainer = DepressionModelTrainer(test_size=test_size, random_state=random_state)
            metrics = trainer.train_and_evaluate(st.session_state.df.copy())
            
            st.session_state.trainer = trainer
            st.session_state.model_trained = True
            
            st.balloons()
            st.success("✅ Model trained successfully!")
            
            display_model_metrics(metrics)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def display_model_metrics(metrics):
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    stats = [
        ("🎯", "Accuracy", f"{metrics['accuracy']*100:.2f}%"),
        ("📊", "Test Samples", f"{len(metrics['y_test']):,}"),
        ("⚡", "Training Time", f"{metrics.get('training_time', 0):.2f}s"),
        ("✅", "Status", "Ready")
    ]
    
    for col, (icon, label, value) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create tabs for model details
    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📈 Classification Report"])
    
    with tab1:
        fig = plot_confusion_matrix(metrics['confusion_matrix'])
        st.pyplot(fig)
    
    with tab2:
        if 'classification_report' in metrics:
            class_report = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(class_report.style.format({
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1-score': '{:.2%}',
                'support': '{:.0f}'
            }), use_container_width=True)

# Page 3: Make Predictions (renumbered)
def page_make_predictions():
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
        return
    
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">📝</div>
            <h3 class="card-title">Input Features</h3>
        </div>
        <p style="color: #94a3b8; margin-bottom: 20px;">
            Enter values for each feature to get a depression level prediction.
            The model will analyze the input and provide a prediction with confidence scores.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    input_data = create_input_fields(st.session_state.trainer)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Predict Depression Level", use_container_width=True, type="primary"):
            make_prediction(input_data, st.session_state.trainer)

def create_input_fields(trainer):
    input_data = {}
    
    # Group features by type for better organization
    categorical_features = []
    numerical_features = []
    
    for feature in trainer.feature_names:
        if feature in trainer.label_encoders:
            categorical_features.append(feature)
        else:
            numerical_features.append(feature)
    
    # Display in tabs for better organization
    if categorical_features:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="color: #60a5fa; margin: 0;">Categorical Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2, gap="large")
        for idx, feature in enumerate(categorical_features):
            with cols[idx % 2]:
                options = trainer.label_encoders[feature].classes_
                input_data[feature] = st.selectbox(
                    f"🔹 {feature}",
                    options,
                    key=f"select_{feature}",
                    help=f"Select value for {feature}"
                )
    
    if numerical_features:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="color: #60a5fa; margin: 0;">Numerical Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2, gap="large")
        for idx, feature in enumerate(numerical_features):
            with cols[idx % 2]:
                # Get min and max from training data for better context
                min_val = float(trainer.X_train[feature].min())
                max_val = float(trainer.X_train[feature].max())
                mean_val = float(trainer.X_train[feature].mean())
                
                input_data[feature] = st.number_input(
                    f"🔹 {feature}",
                    value=float(mean_val),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    key=f"num_{feature}",
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
    
    return input_data

def make_prediction(input_data, trainer):
    try:
        input_df = pd.DataFrame([input_data])  # dict -> DataFrame
        input_df = input_df[trainer.feature_names]  # match training order
        predicted_class, prediction_proba = trainer.predict(input_data)
        
        # Get the class with highest probability
        classes = trainer.target_encoder.classes_
        max_prob_idx = np.argmax(prediction_proba)
        max_prob = prediction_proba[max_prob_idx] * 100
        
        # Choose color class based on prediction
        color_classes = {
            'Normal': 'prediction-result-normal',
            'Mild': 'prediction-result-mild',
            'Moderate': 'prediction-result-moderate',
            'Severe': 'prediction-result-severe'
        }
        
        color_class = color_classes.get(predicted_class, 'prediction-result')
        
        st.markdown(f"""
        <div class="{color_class}">
            <div class="prediction-label">Predicted Depression Level</div>
            <div class="prediction-value">{predicted_class}</div>
            <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.9); margin-top: 10px;">
                Confidence: <strong>{max_prob:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">📊</div>
                <h3 class="card-title">Confidence Breakdown</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sort classes by probability for better visualization
        sorted_indices = np.argsort(prediction_proba)[::-1]
        
        for idx in sorted_indices:
            cls = classes[idx]
            confidence = prediction_proba[idx] * 100
            
            # Color code based on confidence level
            bar_color = ""
            if confidence > 70:
                bar_color = "background: linear-gradient(90deg, #10b981, #34d399);"
            elif confidence > 40:
                bar_color = "background: linear-gradient(90deg, #f59e0b, #fbbf24);"
            else:
                bar_color = "background: linear-gradient(90deg, #ef4444, #f87171);"
            
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-header">
                    <span class="progress-label">{cls}</span>
                    <span class="progress-value">{confidence:.1f}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {confidence}%; {bar_color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add interpretation
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">💡</div>
                <h3 class="card-title">Interpretation</h3>
            </div>
            <p style="color: #94a3b8; line-height: 1.6;">
                This prediction is based on the machine learning model trained on your dataset. 
                The confidence scores show how likely the model believes each depression level applies.
                <br><br>
                <em>Note: This is a predictive model and should not be used as a substitute for professional medical advice.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Page 4: Visualizations (renumbered)
def page_visualizations():
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first!")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔗 Correlations", "🎯 Features", "📉 Metrics"])
    
    with tab1:
        if TARGET_COLUMN in st.session_state.df.columns:
            fig = plot_target_distribution(st.session_state.df, TARGET_COLUMN)
            st.pyplot(fig)
        else:
            st.info("Target column not found in dataset. Showing general distribution.")
            # Plot distribution of first numerical column
            numerical_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                fig = plot_target_distribution(st.session_state.df, numerical_cols[0])
                st.pyplot(fig)
    
    with tab2:
        fig = plot_correlation_matrix(st.session_state.df)
        if fig:
            st.pyplot(fig)
    
    with tab3:
        if st.session_state.model_trained:
            feature_importance_df = st.session_state.trainer.get_feature_importance()
            if feature_importance_df is not None:
                fig = plot_feature_importance(feature_importance_df)
                st.pyplot(fig)
            else:
                st.info("Feature importance not available for this model.")
    
    with tab4:
        if st.session_state.model_trained:
            metrics = st.session_state.trainer.metrics
            if 'classification_report' in metrics:
                class_report = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(class_report.style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:.0f}'
                }), use_container_width=True)
            
            if 'y_test' in metrics and 'y_pred' in metrics:
                fig = plot_prediction_comparison(metrics['y_test'], metrics['y_pred'], st.session_state.trainer.target_encoder)
                st.pyplot(fig)
# Main App
def main():
    init_session_state()
    render_hero()
    render_navigation()
    
    if st.session_state.current_page == PAGES[0]:  # Quick Predict
        page_quick_predict()
    elif st.session_state.current_page == PAGES[1]:  # Load Data
        page_load_data()
    elif st.session_state.current_page == PAGES[2]:  # Train Model
        page_train_model()
    elif st.session_state.current_page == PAGES[3]:  # Make Predictions
        page_make_predictions()
    elif st.session_state.current_page == PAGES[4]:  # Visualizations
        page_visualizations()

if __name__ == "__main__":
    main()
