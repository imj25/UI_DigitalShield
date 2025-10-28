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
import requests
import html

# Add project root to path for RAG system imports
# From UI folder, go up to main project directory
project_root = Path(__file__).parent.parent  # UI -> Digital_Shield1
sys.path.insert(0, str(project_root))

# External API configuration
API_BASE_URL = os.getenv(
    "DIGITAL_SHIELD_API_BASE",
    "https://digitalshield-1023859742049.europe-west1.run.app"
).rstrip("/")
BASE_URL = API_BASE_URL
RAG_PATH = os.getenv("DIGITAL_SHIELD_RAG_PATH", "/rag_chat")
if not RAG_PATH.startswith("/"):
    RAG_PATH = "/" + RAG_PATH
RAG_API_URL = f"{API_BASE_URL}{RAG_PATH}"
RAG_CANDIDATE_PATHS = [
    RAG_PATH,
    "/rag_chat",
    "/rag",
    "/rag/rag_chat",
    "/rag/rag_chat_rag_chat_post",
]

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
    :root {
        /* Core brand colors */
        --primary: #a429aa; /* neon pink */
        --primaryHover: #69196c; /* darker neon pink for hover states */
        --accent: #69196c; /* same as dark neon pink for accents */

        /* Neutral tones */
        --bg: #000000; /* black background */
        --surface: #3f3e3e; /* dark grey surfaces like cards or panels */
        --text: #ffffff; /* white text for contrast */
        --mutedText: #a6a6a6; /* gray for subtitles or secondary info */

        /* Status colors */
        --success: #4caf50; /* green tone */
        --warning: #ffb300; /* amber/yellow tone */
        --danger: #f44336; /* red tone */
        /* Darker shades for gradients */
        --successDark: #2e7d32;
        --warningDark: #ff8f00;
        --dangerDark: #c62828;

        /* Optional gradient */
        --gradient-primary: linear-gradient(135deg, #a429aa, #69196c);
        --gradient-surface-alt: linear-gradient(135deg, #2a2a2a, #101010);
    }

    /* App base */
    .stApp, body { background-color: var(--bg); color: var(--text); }
    /* Main dashboard styling */
    .main-header {
            text-align: center;
        color: var(--primary);
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
            text-align: center;
        color: var(--mutedText);
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
        background-color: var(--surface);
        border-radius: 10px 10px 0 0;
        color: var(--text);
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: var(--text);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--primaryHover);
        color: var(--text);
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
        border: 3px solid var(--primary) !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    div[data-testid="stImage"] img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 3px solid var(--primary) !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    /* Custom chat input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid var(--primary);
    }
    
    .stChatInput > div > div > input:focus {
        border-color: var(--primaryHover);
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.25);
    }
    
    /* Custom pills styling */
    .stPills {
        margin-top: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: var(--text);
        border-radius: 20px;
        border: none;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(31, 78, 121, 0.15);
        margin-top: 0.5rem;
    }
    
    .stButton > button:hover {
        background: var(--primaryHover);
    }
    
    /* Financial Loss Model styling */
    .metric-card {
        background: var(--gradient-primary);
        color: var(--text);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
    }
    
    .metric-card h3 {
        color: var(--text);
        margin: 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        color: var(--text);
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    .prediction-result {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.12), rgba(0,0,0,0) 40%),
            linear-gradient(135deg, var(--danger) 0%, var(--dangerDark) 85%);
        color: var(--text);
        padding: 1.25rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: none;
    }
    
    .prediction-result h2 {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-result p {
        color: var(--text);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    .warning-box {
        background: var(--warning);
        border: none;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: #2c2c2c;
    }
    
    .warning-box strong {
        color: #d63031;
        font-size: 1.1rem;
    }
    
    .success-box {
        background: var(--success);
        color: var(--text);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    
    .section-header {
        color: var(--text);
        font-size: 1.5rem;
            font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary);
    }
    
    .feature-section {
        background: var(--surface);
        color: var(--text);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .feature-section h4 {
        color: var(--text);
        margin: 0 0 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, var(--success), var(--successDark));
        color: var(--text);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, var(--warning), var(--warningDark));
        color: var(--text);
    }
    
    .risk-high {
        background: linear-gradient(135deg, var(--danger), var(--dangerDark));
        color: var(--text);
    }
    
    .risk-critical {
        background: linear-gradient(135deg, var(--danger), var(--dangerDark));
        color: var(--text);
    }
    
    .input-container {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Cybersecurity Information Center styling */
    .card {
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        min-height: 150px;
        color: var(--text);
        background: var(--surface);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 8px;
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 0 35px rgba(0, 0, 0, 0.35);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    .card-icon { display: none; }

    .card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.35;
    }
    
    /* Subtle dark gradients for main page cards */
    .card-blue { background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(0,0,0,0) 35%), linear-gradient(135deg, #2b2b2b 0%, #121212 85%); }
    .card-cyan { background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(0,0,0,0) 35%), linear-gradient(135deg, #343434 0%, #141414 85%); }
    .card-purple { background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(0,0,0,0) 35%), linear-gradient(135deg, #3a3a3a 0%, #0f0f0f 85%); }
    
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
            background: var(--gradient-surface-alt) !important;
            border-radius: 0 0 12px 12px !important;
            padding: 18px 22px !important;
            color: var(--text) !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            margin-bottom: 20px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            animation: fadeIn 0.4s ease-in-out !important;
        }

        .exp-blue .streamlit-expanderHeader { background: var(--gradient-primary) !important; }
        .exp-cyan .streamlit-expanderHeader { background: var(--gradient-primary) !important; }
        .exp-purple .streamlit-expanderHeader { background: var(--gradient-primary) !important; }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-5px);}
            to {opacity: 1; transform: translateY(0);}
        }
        
        /* Unified medium card sizing - smaller */
        .medium-card {
            min-height: 140px;
            padding: 0.9rem !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        /* Unified heading/value styles inside medium cards */
        .medium-card h3 {
            margin: 0;
            text-align: center;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .medium-card .metric-value {
            margin: 0.25rem 0 0 0;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
        }
        
        /* Decorative curved arrows below section heading */
        .arrows-header {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 16px;
            margin: 0 0 8px 0;
        }
        .arrows-heading { margin: 0; color: var(--primary); font-size: 1.75rem; font-weight: 700; }
        .arrows-header a { display: none !important; }
        .arrows-header svg {
            width: 64px; height: 40px;
            stroke: var(--primary);
            stroke-width: 3;
            fill: none;
            opacity: 0.9;
            display: block;
            transform-origin: 50% 50%;
            animation: arrowBounce 1.8s ease-in-out infinite;
            will-change: transform;
        }

        @keyframes arrowBounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(5px); }
        }

        /* Animated ellipsis for welcome text */
        .ellipsis { display: inline-block; margin-left: 4px; }
        .ellipsis span { 
            display: inline-block; 
            animation: dotFade 1.4s infinite ease-in-out; 
            opacity: 0.2; 
        }
        .ellipsis span:nth-child(1) { animation-delay: 0s; }
        .ellipsis span:nth-child(2) { animation-delay: 0.2s; }
        .ellipsis span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotFade {
            0%, 60%, 100% { opacity: 0.2; }
            30% { opacity: 1; }
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
    images_dir = Path(__file__).parent / "images"
    filename_mapping = {
        "welcome": "Welcome.jpg",
        "processing": "ProcessingState.jpg",
        "success": "Welcome.jpg",  # Use welcome for success
        "error": "ErrorState.jpg",
    }
    filename = filename_mapping.get(state, "Welcome.jpg")
    return str((images_dir / filename).resolve())

def call_rag_api(query: str) -> Dict[str, Any]:
    """Call external RAG API with retries/backoff and endpoint fallback, return response."""
    max_attempts = int(os.getenv("DIGITAL_SHIELD_RAG_RETRIES", "3"))
    base_timeout = float(os.getenv("DIGITAL_SHIELD_RAG_TIMEOUT", "45"))
    backoff_seconds = float(os.getenv("DIGITAL_SHIELD_RAG_BACKOFF", "1.5"))

    last_error = None
    # Determine path resolution: reuse previously discovered working path if available
    resolved_path = st.session_state.get("resolved_rag_path")
    paths_to_try = [resolved_path] if resolved_path else []
    # Append configured and common candidates, dedup while preserving order
    for p in RAG_CANDIDATE_PATHS:
        if p and p not in paths_to_try:
            paths_to_try.append(p)

    for attempt in range(1, max_attempts + 1):
        for path in paths_to_try:
            try:
                path_norm = path if path.startswith("/") else "/" + path
                url = f"{API_BASE_URL}{path_norm}"
                resp = requests.post(url, json={"query": query}, timeout=base_timeout)
                if resp.status_code == 404 or resp.status_code == 405:
                    # Not found or method not allowed: try next candidate path
                    last_error = Exception(f"HTTP {resp.status_code} at {url}: {resp.text[:200]}")
                    continue
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
                # Cache the working path
                st.session_state["resolved_rag_path"] = path_norm
                return {
                    "error": False,
                    "response": data.get("response", ""),
                    "suggested_queries": data.get("suggested_queries", []),
                }
            except Exception as e:
                last_error = e
                # try next path; if no more paths, backoff and retry the set
        if attempt < max_attempts:
            time.sleep(backoff_seconds * attempt)
        else:
            break
    return {"error": True, "message": str(last_error) if last_error else "Unknown error"}

# Helper function to generate responses using RAG system with LLM
def generate_response(user_input, avatar_placeholder):
    """Generate intelligent responses using external RAG API."""
    input_lower = user_input.lower()

    # Set state to processing and update avatar
    st.session_state.current_state = "processing"
    avatar_path = get_avatar_for_state(st.session_state.current_state)
    if os.path.exists(avatar_path):
        avatar_placeholder.image(avatar_path, width=150)
    else:
        avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)

    # Use external RAG API
    result = call_rag_api(user_input)
    if not result.get('error'):
        response = result.get('response', '')
        suggested = result.get('suggested_queries', [])
        if suggested:
            response += "\n\n💡 **You might also ask:**"
            for i, suggestion in enumerate(suggested[:3], 1):
                response += f"\n{i}. {suggestion}"

        st.session_state.current_state = "success"
        st.session_state.success_start_time = time.time()
        avatar_path = get_avatar_for_state(st.session_state.current_state)
        if os.path.exists(avatar_path):
            avatar_placeholder.image(avatar_path, width=150)
        else:
            avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
        return response
    else:
        st.session_state.current_state = "error"
        avatar_path = get_avatar_for_state(st.session_state.current_state)
        if os.path.exists(avatar_path):
            avatar_placeholder.image(avatar_path, width=150)
        else:
            avatar_placeholder.markdown("<div style='font-size: 100px; text-align: center;'></div>", unsafe_allow_html=True)
        return f"There was an error contacting the RAG service: {result.get('message', 'Unknown error')}"

# Reusable footer
def render_footer():
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0; color: #555; margin-top: 2rem;">
            <strong>Digital Shield</strong> - Securing Your Digital World with Saudi Heritage of Protection<br/>
            <span style="font-size: 0.95rem; color: #777;">built with <span style="color:#e25555;">❤</span> by Majid, Nawaf, Nouf, Rawaf</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def call_financial_loss_api(features: Dict[str, Any]) -> Dict[str, Any]:
    """Call external Financial Loss API at /predict_financial_loss and return status/data."""
    try:
        # Use flat payload per service contract
        payload = features
        response = requests.post(
            f"{BASE_URL}/predict_financial_loss",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        data = None
        try:
            data = response.json() if response.content else None
        except Exception:
            data = None
        return {"status_code": response.status_code, "data": data, "text": response.text}
    except Exception as e:
        return {"status_code": None, "error": str(e)}

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
        if st.button("Back to Main Dashboard", key="btn_back_rag"):
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
    """Financial Loss Model page with interactive components (via external API)"""
    # Back button at the top
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("Back to Main Dashboard", key="btn_back_financial"):
            st.session_state.current_tab = "Main Dashboard"
            st.rerun()
    
    st.markdown('<h1 class="main-header">💰 Financial Loss Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Financial Risk Assessment for Cybersecurity Incidents</p>', unsafe_allow_html=True)
    
    
    
    # Create input form
    st.markdown('<h2 class="section-header">🔧 Enter Your Incident Details</h2>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-section card-blue"><h4>📊 Essential Information</h4></div>', unsafe_allow_html=True)
            
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
            st.markdown('<div class="feature-section card-cyan"><h4>🎯 Attack & Target Details</h4></div>', unsafe_allow_html=True)
            
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
        
        
    
    # Prediction button
    if st.button("🔮 Predict Financial Loss", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Build features for API (flat JSON)
            api_features = {
                "number_of_affected_users": int(affected_users),
                "data_breach_size_gb": float(data_breach_gb),
                "attack_type": attack_type,
                "target_industry": target_industry,
            }
            api_result = call_financial_loss_api(api_features)

        if api_result.get("status_code") == 200 and api_result.get("data"):
            result = api_result["data"]

            prediction = float(result.get('prediction', 0) or 0)
            severity_label = result.get('severity')
            # Display prediction result and risk assessment side by side
            st.markdown('<h2 class="section-header">💰 Prediction Result</h2>', unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            with col_left:
                # Color the card by magnitude of predicted loss
                if prediction < 10:
                    loss_class = "risk-low"
                elif prediction < 50:
                    loss_class = "risk-medium"
                else:
                    loss_class = "risk-high"
                st.markdown(f"""
                <div class="metric-card medium-card {loss_class}">
                    <h3>Predicted Financial Loss</h3>
                    <p class="metric-value">${prediction:,.2f} Million</p>
                </div>
                """, unsafe_allow_html=True)
            with col_right:
                # Risk level based on API severity
                sev = (severity_label or "").strip().lower()
                if sev == "low":
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                elif sev == "medium":
                    risk_level = "Medium Risk"
                    risk_class = "risk-medium"
                elif sev == "high":
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                elif sev == "critical":
                    risk_level = "Critical Risk"
                    risk_class = "risk-critical"
                else:
                    risk_level = "Unknown Risk"
                    risk_class = "risk-medium"
                st.markdown(f"""
                <div class="metric-card medium-card {risk_class}">
                    <h3>Risk Assessment</h3>
                    <p class=\"metric-value\">{risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
            
            
            
            # Guidance banner with dynamic title (severity only), static body
            sev = (severity_label or '').strip().lower()
            sev_title = (
                'Low' if sev == 'low' else
                'Medium' if sev == 'medium' else
                'High' if sev == 'high' else
                'Critical' if sev == 'critical' else
                'Risk'
            )
            st.markdown(f"""
            <div class="warning-box">
                🕵️‍ <strong>{sev_title} Risk Notice:</strong> This prediction indicates a financial risk. 
                Take appropriate security actions and consult the <span style="background: rgba(164, 41, 170, 0.25); padding: 2px 6px; border-radius: 6px; font-weight: 800; color: #2c2c2c;">Ḥimā chatbot</span> for personalized recommendations.
            </div>
            """, unsafe_allow_html=True)
        else:
            status_code = api_result.get("status_code")
            if status_code is None and api_result.get("error"):
                st.error(f"❌ Error: {api_result.get('error')}")
            else:
                st.error(f"❌ Error: {status_code}")
                st.code(api_result.get("text", ""))
            
    

# Cybersecurity Information Center Page
def cybersecurity_info_page():
    """Cybersecurity Information Center page - Curated reference content"""
    # Back button at the top
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("Back to Main Dashboard", key="btn_back_cyber"):
            st.session_state.current_tab = "Main Dashboard"
            st.rerun()

    # Page header
    st.markdown('<h1 class="main-header">Cybersecurity Information Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Concise definitions, how attacks happen, and practical defenses</p>', unsafe_allow_html=True)

    # Attack Types
    st.markdown('<h2 class="section-header">Attack Types</h2>', unsafe_allow_html=True)
    with st.expander("Phishing – Social-engineering messages that trick users", expanded=False):
        st.markdown(
            """
            - **Definition**: Deceptive emails, links, or messages that trick people into revealing credentials or running malware.
            - **How it happens**: Attacker spoofs the sender or creates fake sites; the user clicks a link or enters credentials.
            """
        )
    with st.expander("DDoS (Distributed Denial of Service) – Flooding a target to make it unavailable.", expanded=False):
        st.markdown(
            """
            - **Definition**: Massive traffic or request floods that overwhelm services or network links.
            - **How it happens**: Botnets or spoofed request floods attack bandwidth or application resources.
            """
        )
    with st.expander("Man-in-the-Middle (MitM) – Intercepting or altering communications.", expanded=False):
        st.markdown(
            """
            - **Definition**: Intercepting or changing messages between two parties to eavesdrop or tamper.
            - **How it happens**: On insecure Wi‑Fi, poorly configured TLS, or compromised routers/switches.
            """
        )
    with st.expander("SQL Injection – Injecting SQL to read/modify databases.", expanded=False):
        st.markdown(
            """
            - **Definition**: Crafted input alters SQL queries to access or modify data.
            - **How it happens**: Poor input validation and string‑concatenated queries in applications.
            """
        )
    with st.expander("Malware – General-purpose malicious software (virus, trojan, RAT).", expanded=False):
        st.markdown(
            """
            - **Definition**: Software designed to harm, steal, or persist (viruses, trojans, remote access tools).
            - **How it happens**: Delivered via attachments, downloads, compromised sites, or removable media.
            """
        )
    with st.expander("Ransomware – Malware that encrypts files and demands ransom.", expanded=False):
        st.markdown(
            """
            - **Definition**: Malware that encrypts data and extorts payment for decryption.
            - **How it happens**: Delivered via phishing, exposed RDP, or malicious downloads.
            """
        )

    # Defense Mechanisms
    st.markdown('<h2 class="section-header">Defense Mechanisms</h2>', unsafe_allow_html=True)
    with st.expander("Firewall – Filters traffic and enforces network rules."):
        st.markdown(
            """
            - **What it is**: Packet/connection filter that controls allowed traffic.
            - **When it helps**: Blocks unwanted ports/services and isolates segments.
            - **Quick steps**: Maintain policy, block unused ports, log alerts, and combine with IDS.
            """
        )
    with st.expander("Intrusion Detection / Prevention (IDS/IPS)"):
        st.markdown(
            """
            - **What it is**: Monitors traffic or hosts for known bad patterns and alerts or blocks.
            - **When it helps**: Early detection of attacks and anomalous behavior.
            - **Quick steps**: Tune signatures, feed alerts into SOC workflows, use alongside EDR.
            """
        )
    with st.expander("Encryption – Protects data confidentiality at rest and in transit."):
        st.markdown(
            """
            - **What it is**: Transforming data so only authorized parties can read it (keys).
            - **When it helps**: Protects data even if storage or network is compromised.
            - **Quick steps**: Use strong algorithms, manage keys securely, encrypt backups.
            """
        )

    # Vulnerability Types
    st.markdown('<h2 class="section-header">Vulnerability Types</h2>', unsafe_allow_html=True)
    with st.expander("Zero‑Day – Vulnerability exploited before a patch exists."):
        st.markdown(
            """
            - **Definition**: A flaw unknown to the vendor and unpatched when exploited.
            """
        )
    with st.expander("Weak Authentication – Easy‑to‑guess or reused credentials."):
        st.markdown(
            """
            - **Definition**: Passwords or authentication methods that attackers can easily bypass.
            """
        )
    with st.expander("Unpatched Software – Known flaws without vendor fixes applied"):
        st.markdown(
            """
            - **Definition**: Systems missing vendor updates that fix vulnerabilities.
            """
        )

    

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
            pass
    except Exception as e:
        pass
    
    # Main title and subtitle
    st.markdown('<h1 class="main-header">Digital Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Financial Risk & Cybersecurity Intelligence</p>', unsafe_allow_html=True)
    
    # Welcome message with typing animation (runs once per session)
    full_welcome_text_plain = (
        "Welcome to Digital Shield - your intelligent dashboard for analyzing cybersecurity data and predicting potential financial losses. "
        "Harness the power of machine learning and AI chat assistance to make informed, data-driven security decisions"
    )
    full_welcome_text_rich = (
        "Welcome to <strong>Digital Shield</strong> - your intelligent dashboard for analyzing cybersecurity data and predicting potential financial losses. "
        "Harness the power of machine learning and AI chat assistance to make informed, data-driven security decisions"
    )
    if "welcome_typed_done" not in st.session_state:
        st.session_state.welcome_typed_done = False

    welcome_placeholder = st.empty()
    if not st.session_state.welcome_typed_done:
        typed = ""
        for ch in full_welcome_text_plain:
            typed += ch
            welcome_placeholder.markdown(
                f"""
                <div style=\"text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;\">
                    {html.escape(typed)}<span style=\"display:inline-block;width:10px;color:#1f4e79;\">|</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(0.012)
        st.session_state.welcome_typed_done = True
        # Render final without cursor
        welcome_placeholder.markdown(
            f"""
            <div style=\"text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;\">
                {full_welcome_text_rich}<span class=\"ellipsis\"><span>.</span><span>.</span><span>.</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        welcome_placeholder.markdown(
            f"""
            <div style=\"text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;\">
                {full_welcome_text_rich}<span class=\"ellipsis\"><span>.</span><span>.</span><span>.</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Feature cards heading flanked by arrows
    st.markdown(
        """
        <div class="arrows-header">
            <svg viewBox="0 0 64 40" xmlns="http://www.w3.org/2000/svg">
                <path d="M2 2 C 20 30, 44 30, 62 2" />
                <path d="M2 12 C 20 34, 44 34, 62 12" />
            </svg>
            <div class="arrows-heading">Choose your Digital Shield</div>
            <svg viewBox="0 0 64 40" xmlns="http://www.w3.org/2000/svg">
                <path d="M2 2 C 20 30, 44 30, 62 2" />
                <path d="M2 12 C 20 34, 44 34, 62 12" />
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card card-blue' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-title'>Financial Loss Model</div>
            <p>Predict potential monetary loss from cybersecurity incidents with precision.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Financial Loss Model", key="btn_financial", use_container_width=True):
            st.session_state.current_tab = "Financial Loss Model"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='card card-cyan' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-title'>Hima Chatbot</div>
            <p>Get instant, AI-driven cyber insights, guidance, and clear explanations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Hima Chatbot", key="btn_rag", use_container_width=True):
            st.session_state.current_tab = "RAG Chatbot"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='card card-purple' style="cursor: pointer; transition: all 0.3s ease;">
            <div class='card-title'>Cybersecurity Information Center</div>
            <p>Access comprehensive information about attacks, defenses, and vulnerabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Information Center", key="btn_info", use_container_width=True):
            st.session_state.current_tab = "Cybersecurity Information Center"
            st.rerun()
    
    # Footer
    render_footer()

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