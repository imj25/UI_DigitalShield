import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import base64
from io import BytesIO

# Add project root to path for RAG system imports
# From UI folder, go up to main project directory
project_root = Path(__file__).parent.parent  # UI -> Digital_Shield1
sys.path.insert(0, str(project_root))

# Import RAG system
try:
    from Digital_Shield_Packages.RAG.main import RAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    st.warning(f"RAG system not available: {e}")
    RAG_AVAILABLE = False

# Import the financial loss model
try:
    from Digital_Shield_Packages.ML.fanancial_loss_model import ModelSaver
    MODEL_AVAILABLE = True
except ImportError as e:
    st.warning(f"Financial loss model not available: {e}")
    MODEL_AVAILABLE = False

# Configure the page
st.set_page_config(
    page_title="Digital Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the entire application
st.markdown("""
    <style>
    /* Main dashboard styling */
    .main-header {
            text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
            text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: #1f4e79;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2c5aa0;
        color: white;
    }
    
    /* Avatar container styling for RAG chatbot */
    .avatar-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 1rem !important;
        width: 100% !important;
    }
    
    .avatar-container img,
    .stImage img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 3px solid #1f4e79 !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    div[data-testid="stImage"] img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 3px solid #1f4e79 !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    /* Custom chat input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #1f4e79;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #2c5aa0;
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.25);
    }
    
    /* Custom pills styling */
    .stPills {
        margin-top: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #2c5aa0;
    }
    
    /* Financial Loss Model styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
    }
    
    .metric-card h3 {
        color: white;
        margin: 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: none;
    }
    
    .prediction-result h2 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-result p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        border: none;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(254, 202, 87, 0.3);
        color: #2c2c2c;
    }
    
    .warning-box strong {
        color: #d63031;
        font-size: 1.1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    
    .section-header {
        color: #2d3436;
        font-size: 1.5rem;
            font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #74b9ff;
    }
    
    .feature-section {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .feature-section h4 {
        color: white;
        margin: 0 0 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
    }
    
    .input-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Cybersecurity Information Center styling */
    .card {
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        height: 230px;
        color: #fdf6e6;
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 0 35px rgba(255, 255, 255, 0.15);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .card-blue {
        background: linear-gradient(135deg, #1f3b73, #2a5298);
    }
    
    .card-cyan {
        background: linear-gradient(135deg, #136a8a, #267871);
    }
    
    .card-purple {
        background: linear-gradient(135deg, #42275a, #734b6d);
    }
    
    /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: bold !important;
            font-size: 1.05rem !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            margin-bottom: 10px !important;
            color: #fdf6e6 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            border: none !important;
        }
    
        .streamlit-expanderHeader:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.15) !important;
        }
    
        .streamlit-expander {
            border: none !important;
            box-shadow: none !important;
        }
    
        .streamlit-expanderContent {
            background-color: #141a2b !important;
            border-radius: 0 0 12px 12px !important;
            padding: 18px 22px !important;
            color: #dcdcdc !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            margin-bottom: 20px !important;
            animation: fadeIn 0.4s ease-in-out !important;
        }

        .exp-blue .streamlit-expanderHeader {
            background: linear-gradient(135deg, #1f3b73, #2a5298) !important;
        }
    
        .exp-cyan .streamlit-expanderHeader {
            background: linear-gradient(135deg, #136a8a, #267871) !important;
        }
    
        .exp-purple .streamlit-expanderHeader {
            background: linear-gradient(135deg, #42275a, #734b6d) !important;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-5px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_state" not in st.session_state:
    st.session_state.current_state = "welcome"

if "success_start_time" not in st.session_state:
    st.session_state.success_start_time = None

# Function to get avatar based on current state
def get_avatar_for_state(state):
    """Get the appropriate avatar image based on the current state"""
    avatar_mapping = {
        "welcome": "../Digital_Shield_Avatars/Welcome.jpg",
        "processing": "../Digital_Shield_Avatars/Processing State.jpg",
        "success": "../Digital_Shield_Avatars/Welcome.jpg",  # Use welcome for success
        "error": "../Digital_Shield_Avatars/Error State.jpg"
    }
    return avatar_mapping.get(state, "../Digital_Shield_Avatars/Welcome.jpg")

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    """Get or create the RAG system instance (cached for performance)"""
    if RAG_AVAILABLE:
        return RAGSystem()
    return None

def initialize_rag_system():
    """Initialize the RAG system silently in the background"""
    if RAG_AVAILABLE and 'rag_system' not in st.session_state:
        # Initialize silently without showing spinner to users
        rag_system = get_rag_system()
        if rag_system:
            success = rag_system.initialize()
            if success:
                st.session_state.rag_system = rag_system
                st.session_state.rag_ready = True
            else:
                st.session_state.rag_ready = False
        else:
            st.session_state.rag_ready = False

    return st.session_state.get('rag_system', None), st.session_state.get('rag_ready', False)

# Helper function to generate responses using RAG system with LLM
def generate_response(user_input, avatar_placeholder):
    """Generate intelligent responses using RAG system with LLM fallback"""
    input_lower = user_input.lower()

    # Set state to processing and update avatar
    st.session_state.current_state = "processing"
    avatar_path = get_avatar_for_state(st.session_state.current_state)
    if os.path.exists(avatar_path):
        avatar_placeholder.image(avatar_path, width=150)
    else:
        avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)

    # Try to use RAG system first
    rag_system, rag_ready = initialize_rag_system()

    if rag_ready and rag_system:
        try:
            # Use RAG system to get intelligent response
            result = rag_system.query(
                user_input,
                top_k=20,
                similarity_threshold=0.7
            )

            if not result.get('error'):
                response = result.get('response', '')

                # Add suggested queries if available
                suggested = result.get('suggested_queries', [])
                if suggested:
                    response += "\n\n💡 **You might also ask:**"
                    for i, suggestion in enumerate(suggested[:3], 1):
                        response += f"\n{i}. {suggestion}"

                # Set state to success and update avatar
                st.session_state.current_state = "success"
                st.session_state.success_start_time = time.time()  # Start the success timer
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=150)
                else:
                    avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
                
                return response
            else:
                # RAG system error, fall back to hardcoded responses
                st.session_state.current_state = "error"
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=150)
                else:
                    avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
                st.warning("RAG system encountered an error, using fallback responses.")

        except Exception as e:
            st.session_state.current_state = "error"
            avatar_path = get_avatar_for_state(st.session_state.current_state)
            if os.path.exists(avatar_path):
                avatar_placeholder.image(avatar_path, width=150)
            else:
                avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
            st.warning(f"RAG system error: {e}. Using fallback responses.")

    # If RAG system is not available, return a simple message
    st.session_state.current_state = "error"
    avatar_path = get_avatar_for_state(st.session_state.current_state)
    if os.path.exists(avatar_path):
        avatar_placeholder.image(avatar_path, width=150)
    else:
        avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
    return f"""I apologize, but I'm currently unable to process your request: "{user_input}"
Please try again later or contact support for assistance."""

# Financial Loss Model Functions
def load_model():
    """Load the trained financial loss model"""
    try:
        # Try multiple possible paths for the model
        possible_paths = [
            project_root / "Digital_Shield_Packages" / "models" / "financial_loss_xgboost.pkl",
            project_root / "models" / "financial_loss_xgboost.pkl",
            Path("Digital_Shield_Packages/models/financial_loss_xgboost.pkl"),
            Path("models/financial_loss_xgboost.pkl"),
            Path("../Digital_Shield_Packages/models/financial_loss_xgboost.pkl"),
            Path("../../Digital_Shield_Packages/models/financial_loss_xgboost.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if model_path is None:
            st.error(f"Model file not found. Please ensure the model file exists in the correct location.")
            return None
        
        model_artifact = ModelSaver.load_model(model_path)
        return model_artifact
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_smart_defaults(attack_type, target_industry, affected_users, data_breach_gb):
    """Calculate smart defaults based on user inputs"""
    
    # Resolution time based on attack type and severity (in hours)
    base_resolution_times = {
        'DDoS': 8,                     # 8 hours - usually resolved quickly
        'Malware': 24,                 # 1 day - requires cleanup
        'Man-in-the-middle': 12,       # 12 hours - network issue
        'Phishing': 6,                 # 6 hours - user training/notification
        'Ransomware': 48,              # 2 days - complex recovery
        'SQL Injection': 18            # 18 hours - database fix
    }
    
    # Calculate resolution time based on severity
    base_time = base_resolution_times.get(attack_type, 24)
    
    # Adjust resolution time based on affected users and data breach size
    if affected_users >= 1000000 or data_breach_gb >= 1000:
        resolution_time = base_time * 2.5  # Major incidents take longer
    elif affected_users >= 100000 or data_breach_gb >= 100:
        resolution_time = base_time * 1.8  # Medium incidents
    elif affected_users >= 10000 or data_breach_gb >= 10:
        resolution_time = base_time * 1.3  # Small incidents
    else:
        resolution_time = base_time  # Minimal impact
    
    # Cap resolution time at reasonable limits
    resolution_time = min(resolution_time, 168)  # Max 1 week
    
    # Comprehensive vulnerability mapping with multiple options per attack type
    vulnerability_options = {
        'DDoS': ['Unpatched Software', 'Weak Passwords', 'Zero Day'],
        'Malware': ['Social Engineering', 'Unpatched Software', 'Weak Passwords'],
        'Man-in-the-middle': ['Weak Passwords', 'Unpatched Software', 'Social Engineering'],
        'Phishing': ['Social Engineering', 'Weak Passwords', 'Unpatched Software'],
        'Ransomware': ['Social Engineering', 'Unpatched Software', 'Weak Passwords'],
        'SQL Injection': ['Unpatched Software', 'Weak Passwords', 'Zero Day']
    }
    
    # Comprehensive defense mechanism mapping with multiple options per attack type
    defense_options = {
        'DDoS': ['Firewall', 'VPN', 'AI-based Detection'],
        'Malware': ['Antivirus', 'AI-based Detection', 'Encryption'],
        'Man-in-the-middle': ['Encryption', 'VPN', 'Firewall'],
        'Phishing': ['AI-based Detection', 'Antivirus', 'Encryption'],
        'Ransomware': ['Encryption', 'AI-based Detection', 'Antivirus'],
        'SQL Injection': ['Firewall', 'Encryption', 'AI-based Detection']
    }
    
    # Calculate severity based on multiple factors (more realistic conditions)
    severity_score = 0
    
    # User impact scoring
    if affected_users >= 10000000:  # 10M+ users
        severity_score += 4
    elif affected_users >= 1000000:  # 1M+ users
        severity_score += 3
    elif affected_users >= 100000:   # 100K+ users
        severity_score += 2
    elif affected_users >= 10000:   # 10K+ users
        severity_score += 1
    
    # Data breach impact scoring
    if data_breach_gb >= 10000:      # 10TB+ breach
        severity_score += 4
    elif data_breach_gb >= 1000:     # 1TB+ breach
        severity_score += 3
    elif data_breach_gb >= 100:      # 100GB+ breach
        severity_score += 2
    elif data_breach_gb >= 10:       # 10GB+ breach
        severity_score += 1
    
    # Attack type severity multiplier
    attack_severity_multiplier = {
        'Ransomware': 1.5,           # Most severe - business disruption
        'SQL Injection': 1.3,        # High - data theft
        'Malware': 1.2,              # Medium-high - system compromise
        'Man-in-the-middle': 1.1,    # Medium - data interception
        'Phishing': 1.0,             # Medium - social engineering
        'DDoS': 0.8                  # Lower - service disruption only
    }
    
    # Industry risk multiplier
    industry_risk_multiplier = {
        'Banking': 1.4,              # High value data
        'Healthcare': 1.3,           # Sensitive personal data
        'Government': 1.3,           # National security implications
        'IT': 1.2,                   # Technology sector
        'Telecommunications': 1.1,   # Infrastructure
        'Retail': 1.0,               # Standard risk
        'Education': 0.9             # Lower immediate financial impact
    }
    
    # Apply multipliers
    severity_score *= attack_severity_multiplier.get(attack_type, 1.0)
    severity_score *= industry_risk_multiplier.get(target_industry, 1.0)
    
    # Determine final severity
    if severity_score >= 6:
        severity = 'Critical'
    elif severity_score >= 3:
        severity = 'Medium'
    else:
        severity = 'Low'
    
    # Select vulnerability and defense based on attack type, severity, and industry
    vulnerability_type = vulnerability_options.get(attack_type, ['Weak Passwords', 'Social Engineering', 'Unpatched Software'])
    defense_mechanism = defense_options.get(attack_type, ['Antivirus', 'Firewall', 'Encryption'])
    
    # Industry-specific adjustments
    industry_vulnerability_preferences = {
        'Banking': ['Weak Passwords', 'Social Engineering', 'Unpatched Software'],  # Banking focuses on auth
        'Healthcare': ['Unpatched Software', 'Social Engineering', 'Weak Passwords'],  # Healthcare has legacy systems
        'Government': ['Unpatched Software', 'Zero Day', 'Social Engineering'],  # Gov has complex systems
        'IT': ['Zero Day', 'Unpatched Software', 'Social Engineering'],  # IT companies face advanced threats
        'Telecommunications': ['Unpatched Software', 'Weak Passwords', 'Social Engineering'],  # Telecom infrastructure
        'Retail': ['Social Engineering', 'Weak Passwords', 'Unpatched Software'],  # Retail focuses on human factors
        'Education': ['Social Engineering', 'Weak Passwords', 'Unpatched Software']  # Education has user training issues
    }
    
    industry_defense_preferences = {
        'Banking': ['Encryption', 'AI-based Detection', 'Firewall'],  # Banking prioritizes data protection
        'Healthcare': ['Encryption', 'Antivirus', 'AI-based Detection'],  # Healthcare needs compliance
        'Government': ['Firewall', 'Encryption', 'VPN'],  # Government needs network security
        'IT': ['AI-based Detection', 'Encryption', 'Firewall'],  # IT companies use advanced tech
        'Telecommunications': ['Firewall', 'VPN', 'Encryption'],  # Telecom needs network protection
        'Retail': ['Antivirus', 'AI-based Detection', 'Encryption'],  # Retail balances cost and security
        'Education': ['Antivirus', 'Firewall', 'Encryption']  # Education uses standard protections
    }
    
    # Adjust options based on industry
    industry_vuln_pref = industry_vulnerability_preferences.get(target_industry, vulnerability_type)
    industry_def_pref = industry_defense_preferences.get(target_industry, defense_mechanism)
    
    # Choose based on severity - higher severity gets more appropriate options for the industry
    if severity == 'Critical':
        # For critical incidents, use the most effective option for this industry
        vulnerability_type = industry_vuln_pref[0]
        defense_mechanism = industry_def_pref[0]
    elif severity == 'Medium':
        # For medium incidents, use secondary option
        vulnerability_type = industry_vuln_pref[1] if len(industry_vuln_pref) > 1 else industry_vuln_pref[0]
        defense_mechanism = industry_def_pref[1] if len(industry_def_pref) > 1 else industry_def_pref[0]
    else:  # Low severity
        # For low incidents, use tertiary option or fallback
        vulnerability_type = industry_vuln_pref[2] if len(industry_vuln_pref) > 2 else industry_vuln_pref[0]
        defense_mechanism = industry_def_pref[2] if len(industry_def_pref) > 2 else industry_def_pref[0]
    
    return {
        'year': 2024,
        'incident resolution time (in hours)': resolution_time,
        'country': 'UK',
        'security vulnerability type': vulnerability_type,
        'defense mechanism used': defense_mechanism,
        'severity_kmeans': severity
    }

def engineer_features(df):
    """Apply the same feature engineering as in training"""
    df = df.copy()
    
    # Log transformations
    if "number of affected users" in df.columns:
        df["log_users"] = np.log1p(df["number of affected users"].fillna(0))
    
    if "data breach in gb" in df.columns:
        df["log_breach"] = np.log1p(df["data breach in gb"].fillna(0))
    
    if "incident resolution time (in hours)" in df.columns:
        df["log_resolution_time"] = np.log1p(df["incident resolution time (in hours)"].fillna(0))
    
    # Interaction features
    if "number of affected users" in df.columns and "data breach in gb" in df.columns:
        df["impact_index"] = df["number of affected users"] * np.log1p(df["data breach in gb"].fillna(0))
    
    if "number of affected users" in df.columns and "incident resolution time (in hours)" in df.columns:
        df["users_per_hour"] = df["number of affected users"] / (1.0 + df["incident resolution time (in hours)"].fillna(0))
    
    # Time features
    if "year" in df.columns:
        df["years_since_2010"] = pd.to_numeric(df["year"], errors="coerce") - 2010
    
    # Complex features
    if "impact_index" in df.columns and "users_per_hour" in df.columns:
        df["severity_ratio"] = df["impact_index"] / (1.0 + df["users_per_hour"])
    
    if "log_users" in df.columns and "log_breach" in df.columns:
        df["complexity"] = df["log_users"] * df["log_breach"]
    
    # Remove infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

def preprocess_input_data(input_data):
    """Preprocess input data to match training format"""
    # Create a DataFrame with the input data
    df = pd.DataFrame([input_data])
    
    # First, apply feature engineering (same as in training)
    df = engineer_features(df)
    
    # One-hot encode categorical variables
    categorical_columns = [
        'country', 'attack type', 'target industry', 
        'security vulnerability type', 'defense mechanism used'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            # Create dummy variables for each category
            for category in df[col].unique():
                if pd.notna(category):
                    dummy_col = f"{col}_{category.lower().replace(' ', '_').replace('-', '_')}"
                    df[dummy_col] = (df[col] == category).astype(int)
    
    # Keep severity_kmeans as a single categorical column (not dummy encoded)
    # The model expects it as a single column with values like "Critical", "Medium", "Low"
    
    # Drop other original categorical columns
    df = df.drop(columns=categorical_columns)
    
    # Add all possible dummy columns with 0 values
    all_dummy_columns = [
        'country_australia', 'country_brazil', 'country_china', 'country_france',
        'country_germany', 'country_india', 'country_japan', 'country_russia',
        'country_uk', 'country_usa',
        'attack type_ddos', 'attack type_malware', 'attack type_man-in-the-middle',
        'attack type_phishing', 'attack type_ransomware', 'attack type_sql injection',
        'target industry_banking', 'target industry_education', 'target industry_government',
        'target industry_healthcare', 'target industry_it', 'target industry_retail',
        'target industry_telecommunications',
        'security vulnerability type_social engineering', 'security vulnerability type_unpatched software',
        'security vulnerability type_weak passwords', 'security vulnerability type_zero day',
        'defense mechanism used_ai based detection', 'defense mechanism used_antivirus',
        'defense mechanism used_encryption', 'defense mechanism used_firewall',
        'defense mechanism used_vpn'
    ]
    
    for col in all_dummy_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure all columns are in the correct order
    df = df.reindex(columns=sorted(df.columns), fill_value=0)
    
    return df

def make_prediction(model_artifact, input_data):
    """Make prediction using the loaded model"""
    try:
        # Preprocess input data
        processed_data = preprocess_input_data(input_data)
        
        # Get model components
        bst = model_artifact["model"]
        preprocessor = model_artifact["preprocessor"]
        feature_names = model_artifact["feature_names"]
        
        # Transform data using the preprocessor
        X_processed = preprocessor.transform(processed_data)
        
        # Create DMatrix for XGBoost
        from xgboost import DMatrix
        dx = DMatrix(X_processed, feature_names=feature_names)
        
        # Make prediction
        pred_log = bst.predict(dx)
        prediction = np.expm1(pred_log)[0]  # Inverse log transform
        
        return prediction, processed_data
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# RAG Chatbot Page
def rag_chatbot_page():
    """RAG Chatbot page with original design"""
    # Back button at the top
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("🏠 Back to Main Dashboard", key="btn_back_rag"):
            st.session_state.current_tab = "Main Dashboard"
            st.rerun()
    
    # Centered avatar and title
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Avatar container with placeholder for dynamic updates
        st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
        avatar_placeholder = st.empty()
        
        # Handle success state timing
        if st.session_state.current_state == "success":
            if st.session_state.success_start_time is None:
                st.session_state.success_start_time = time.time()
            elif time.time() - st.session_state.success_start_time > 3.0:  # 3 second delay
                st.session_state.current_state = "welcome"
                st.session_state.success_start_time = None
        
        avatar_path = get_avatar_for_state(st.session_state.current_state)
        if os.path.exists(avatar_path):
            avatar_placeholder.image(avatar_path, width=120)
        else:
            avatar_placeholder.markdown("<div style='font-size: 80px; text-align: center;'></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Centered title and caption
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.title("Ḥimā - Securing Your Digital World", anchor=False)
        st.caption("With Saudi heritage of protection")
        st.markdown('</div>', unsafe_allow_html=True)

    # Title row for restart button
    title_row = st.container(horizontal=True, vertical_alignment="bottom")

    # Check if user has interacted yet
    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )
    
    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
    )
    
    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )
    
    has_message_history = (
        "messages" in st.session_state and len(st.session_state.messages) > 0
    )

    # Show welcome screen when user hasn't asked a question yet
    if not user_first_interaction and not has_message_history:
        st.session_state.messages = []

        with st.container():
            st.chat_input("🔍 Ask about cyber threats...", key="initial_question")

            # Example suggestions
            suggestions = [
                "How to protect my bank from cyber threats?",
                "What are the latest phishing techniques?",
                "How to secure my business from ransomware?",
                "What are the top cybersecurity trends in 2024?",
                "How to identify fake banking emails?"
            ]
            
            selected_suggestion = st.pills(
                label="💡 Example Questions",
                label_visibility="collapsed",
                options=suggestions,
                key="selected_suggestion",
            )

        st.stop()

    # Show chat input at the bottom when a question has been asked
    user_message = st.chat_input("🔍 Ask a follow-up question...")

    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
        if user_just_clicked_suggestion:
            user_message = st.session_state.selected_suggestion

    # Clear conversation button
    with title_row:
        def clear_conversation():
            st.session_state.messages = []
            st.session_state.initial_question = None
            st.session_state.selected_suggestion = None
            st.session_state.current_state = "welcome"

        st.button("🔄 Restart", on_click=clear_conversation)

    # Display chat messages from history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # Fix ghost message bug
            st.markdown(message["content"])

    if user_message:
        # When the user posts a message...
        
        # Display user message
        with st.chat_message("user"):
            st.text(user_message)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔒 Analyzing your security question..."):
                # Update avatar to processing state
                st.session_state.current_state = "processing"
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=120)
                else:
                    avatar_placeholder.markdown("<div style='font-size: 80px; text-align: center;'>🔒</div>", unsafe_allow_html=True)
                
                # Generate response
                response = generate_response(user_message, avatar_placeholder)

            # Put everything after the spinner in a container to fix ghost message bug
            with st.container():
                # Display the response
                st.markdown(response)

                # Add messages to chat history
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": response})

# Financial Loss Model Page
def financial_loss_page():
    """Financial Loss Model page with interactive components"""
    # Back button at the top
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("🏠 Back to Main Dashboard", key="btn_back_financial"):
            st.session_state.current_tab = "Main Dashboard"
            st.rerun()
    
    st.markdown('<h1 class="main-header">💰 Financial Loss Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Financial Risk Assessment for Cybersecurity Incidents</p>', unsafe_allow_html=True)
    
    if not MODEL_AVAILABLE:
        st.error("Financial loss model is not available. Please check the import paths.")
        return
    
    # Load model
    with st.spinner("Loading financial loss model..."):
        model_artifact = load_model()
    
    if model_artifact is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    st.markdown('<div class="success-box">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
    
    # Display model metrics
    if "metrics" in model_artifact:
        metrics = model_artifact["metrics"]
        st.markdown('<h2 class="section-header">📊 Model Performance</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Score</h3>
                <p>{metrics.get('R2', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE</h3>
                <p>{metrics.get('MAE', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <p>{metrics.get('RMSE', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>sMAPE</h3>
                <p>{metrics.get('sMAPE', 0):.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Create input form
    st.markdown('<h2 class="section-header">🔧 Enter Your Incident Details</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-section"><h4>📊 Essential Information</h4></div>', unsafe_allow_html=True)
            
            # Essential inputs only
            affected_users = st.number_input(
                "👥 Number of Affected Users", 
                min_value=1, 
                value=100000,
                help="How many users were affected by the incident?"
            )
            
            data_breach_gb = st.number_input(
                "💾 Data Breach Size (GB)", 
                min_value=0.0, 
                value=50.0,
                help="How much data was compromised in gigabytes?"
            )
        
        with col2:
            st.markdown('<div class="feature-section"><h4>🎯 Attack & Target Details</h4></div>', unsafe_allow_html=True)
            
            attack_type = st.selectbox(
                "⚔️ Attack Type",
                ["DDoS", "Malware", "Man-in-the-middle", "Phishing", "Ransomware", "SQL Injection"],
                help="What type of cyber attack occurred?"
            )
            
            target_industry = st.selectbox(
                "🏢 Target Industry",
                ["Banking", "Education", "Government", "Healthcare", "IT", "Retail", "Telecommunications"],
                help="Which industry was targeted?"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 Predict Financial Loss", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Get smart defaults
            smart_defaults = get_smart_defaults(attack_type, target_industry, affected_users, data_breach_gb)
            
            # Prepare complete input data
            input_data = {
                'year': smart_defaults['year'],
                'number of affected users': affected_users,
                'incident resolution time (in hours)': smart_defaults['incident resolution time (in hours)'],
                'data breach in gb': data_breach_gb,
                'country': smart_defaults['country'],
                'attack type': attack_type,
                'target industry': target_industry,
                'security vulnerability type': smart_defaults['security vulnerability type'],
                'defense mechanism used': smart_defaults['defense mechanism used'],
                'severity_kmeans': smart_defaults['severity_kmeans']
            }
            
            prediction, processed_data = make_prediction(model_artifact, input_data)
        
        if prediction is not None:
            # Display prediction result
            st.markdown('<h2 class="section-header">💰 Prediction Result</h2>', unsafe_allow_html=True)
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-result">
                <h2>${prediction:,.2f} Million</h2>
                <p>Predicted Financial Loss</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment and confidence interval
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level based on prediction
                if prediction < 10:
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                elif prediction < 50:
                    risk_level = "Medium Risk"
                    risk_class = "risk-medium"
                else:
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3>{risk_level}</h3>
                    <p>Risk Assessment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Confidence interval (simplified)
                confidence_lower = prediction * 0.8
                confidence_upper = prediction * 1.2
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence Interval</h3>
                    <p>${confidence_lower:,.1f}M - ${confidence_upper:,.1f}M</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.markdown('<h2 class="section-header">📊 Prediction Visualization</h2>', unsafe_allow_html=True)
            
            # Create a bar chart showing the prediction
            fig = go.Figure(data=[
                go.Bar(
                    x=['Predicted Loss'],
                    y=[prediction],
                    marker_color='#ff6b6b',
                    text=[f'${prediction:,.1f}M'],
                    textposition='auto',
                    textfont=dict(size=20, color='white'),
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text="Financial Loss Prediction",
                    font=dict(size=24, color='#2d3436')
                ),
                yaxis_title="Loss (Million $)",
                showlegend=False,
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=16)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Warning for high predictions
            if prediction > 100:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>High Risk Warning:</strong> This prediction indicates a very high financial loss. 
                    Consider immediate security measures and incident response protocols.
                </div>
                """, unsafe_allow_html=True)
            elif prediction > 50:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>Medium Risk Warning:</strong> This prediction indicates a significant financial loss. 
                    Review your security measures and incident response capabilities.
                </div>
                """, unsafe_allow_html=True)
            

# Cybersecurity Information Center Page
def cybersecurity_info_page():
    """Cybersecurity Information Center page - Empty placeholder"""
    # Back button at the top
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("🏠 Back to Main Dashboard", key="btn_back_cyber"):
            st.session_state.current_tab = "Main Dashboard"
            st.rerun()
    
    # Empty page content
    st.markdown('<h1 class="main-header">ℹ️ Cybersecurity Information Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">This page is currently empty</p>', unsafe_allow_html=True)

# Main Dashboard Page
def main_dashboard_page():
    """Main dashboard landing page with Digital Shield branding"""
    
    # Load and display the Digital Shield logo
    try:
        logo_path = "lmags/p2.jpg"
        if os.path.exists(logo_path):
            # Convert image to base64 for display
            with open(logo_path, "rb") as f:
                logo_data = f.read()
            logo_base64 = base64.b64encode(logo_data).decode()
            
            # Display logo with styling - Notion banner size
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 3rem;">
                <img src="data:image/jpeg;base64,{logo_base64}" 
                     style="width: 100%; max-width: 1200px; height: 300px; object-fit: cover; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.15); margin: 0 auto;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<h1 class="main-header">🛡️ Ḥimā - Digital Shield</h1>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<h1 class="main-header">🛡️ Ḥimā - Digital Shield</h1>', unsafe_allow_html=True)
    
    # Main title and subtitle
    st.markdown('<h1 class="main-header">Digital Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Financial Risk & Cybersecurity Intelligence</p>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
        Welcome to <strong>Digital Shield</strong> — your intelligent dashboard for analyzing cybersecurity data and predicting potential financial losses.
        Harness the power of machine learning and AI chat assistance to make informed, data-driven security decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown('<h2 style="text-align: center; color: #1f4e79; margin-bottom: 2rem;">Choose Your Tool</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card card-blue' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-icon'>💹</div>
            <div class='card-title'>Financial Loss Model</div>
            <p>Predict potential monetary loss from cybersecurity incidents with precision.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💰 Go to Financial Loss Model", key="btn_financial", use_container_width=True):
            st.session_state.current_tab = "Financial Loss Model"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='card card-cyan' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-icon'>🤖</div>
            <div class='card-title'>RAG Chatbot</div>
            <p>Chat with an intelligent assistant to gain instant insights and explanations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Go to RAG Chatbot", key="btn_rag", use_container_width=True):
            st.session_state.current_tab = "RAG Chatbot"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='card card-purple' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-icon'>ℹ️</div>
            <div class='card-title'>Cybersecurity Information Center</div>
            <p>Access comprehensive information about attacks, defenses, and vulnerabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ℹ️ Go to Information Center", key="btn_info", use_container_width=True):
            st.session_state.current_tab = "Cybersecurity Information Center"
            st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 1rem; margin: 3rem 0;">
        <h2 style="color: white; margin: 0;">🔒 Digital Shield</h2>
        <p style="color: #f0f0f0; margin: 0.5rem 0 0 0;">Securing Your Digital World with Saudi Heritage of Protection</p>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Initialize session state for current tab
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Main Dashboard"
    
    # Check if user wants to go to a specific tab
    if st.session_state.current_tab == "Main Dashboard":
        main_dashboard_page()
    else:
        # Show the selected tab
        if st.session_state.current_tab == "Financial Loss Model":
            financial_loss_page()
        elif st.session_state.current_tab == "RAG Chatbot":
            rag_chatbot_page()
        elif st.session_state.current_tab == "Cybersecurity Information Center":
            cybersecurity_info_page()

if __name__ == "__main__":
    main()