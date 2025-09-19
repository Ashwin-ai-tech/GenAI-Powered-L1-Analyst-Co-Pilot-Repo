import streamlit as st
from rag_retrieval import query_rag, get_last_turns, get_conversation_history, call_gemini_raw
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random

# -------------------------
# Enhanced Feedback Database Functions
# -------------------------
class FeedbackDB:
    def __init__(self):
        if "feedback_database" not in st.session_state:
            st.session_state.feedback_database = {
                "feedbacks": [],
                "analytics": {
                    "total_feedback": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "feedback_trend": [],
                    "category_distribution": {}
                }
            }
    
    def store_feedback(self, feedback_data):
        feedback_data['id'] = len(st.session_state.feedback_database["feedbacks"]) + 1
        feedback_data['status'] = 'new'  # new, reviewed, action_taken, closed
        feedback_data['priority'] = self._calculate_priority(feedback_data)
        feedback_data['category'] = self._categorize_feedback(feedback_data)
        
        st.session_state.feedback_database["feedbacks"].append(feedback_data)
        self._update_analytics(feedback_data)
        return feedback_data['id']
    
    def _calculate_priority(self, feedback_data):
        # Priority calculation based on multiple factors
        priority_score = 0
        
        # Negative feedback gets higher priority
        if feedback_data['feedback_action'] == 'thumbs_down':
            priority_score += 3
        
        # Feedback with detailed comments gets higher priority
        if feedback_data['feedback_text'] and len(feedback_data['feedback_text']) > 20:
            priority_score += 2
        
        # Recent feedback gets slightly higher priority
        feedback_time = datetime.fromisoformat(feedback_data['timestamp'])
        if datetime.now() - feedback_time < timedelta(hours=24):
            priority_score += 1
        
        if priority_score >= 3:
            return 'high'
        elif priority_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_feedback(self, feedback_data):
        text = (feedback_data['user_input'] + " " + 
                feedback_data['assistant_answer'] + " " + 
                feedback_data.get('feedback_text', '')).lower()
        
        categories = {
            'accuracy': ['wrong', 'incorrect', 'error', 'mistake', 'not correct'],
            'completeness': ['incomplete', 'missing', 'partial', 'half'],
            'relevance': ['irrelevant', 'off-topic', 'unrelated'],
            'clarity': ['confusing', 'unclear', 'vague', 'hard to understand'],
            'technical': ['technical', 'code', 'implementation', 'bug'],
            'knowledge_base': ['kb', 'knowledge base', 'documentation'],
            'response_time': ['slow', 'fast', 'response time', 'waiting'],
            'general': ['good', 'great', 'helpful', 'useful', 'bad', 'poor']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'uncategorized'
    
    def _update_analytics(self, feedback_data):
        analytics = st.session_state.feedback_database["analytics"]
        analytics['total_feedback'] += 1
        
        if feedback_data['feedback_action'] == 'thumbs_up':
            analytics['positive_count'] += 1
        else:
            analytics['negative_count'] += 1
        
        # Update trend data (last 7 days)
        today = datetime.now().date()
        feedback_date = datetime.fromisoformat(feedback_data['timestamp']).date()
        
        if feedback_date >= today - timedelta(days=7):
            date_str = feedback_date.isoformat()
            if date_str not in [item['date'] for item in analytics['feedback_trend']]:
                analytics['feedback_trend'].append({
                    'date': date_str,
                    'positive': 0,
                    'negative': 0,
                    'total': 0
                })
            
            for trend_item in analytics['feedback_trend']:
                if trend_item['date'] == date_str:
                    trend_item['total'] += 1
                    if feedback_data['feedback_action'] == 'thumbs_up':
                        trend_item['positive'] += 1
                    else:
                        trend_item['negative'] += 1
        
        # Update category distribution
        category = feedback_data.get('category', 'uncategorized')
        if category not in analytics['category_distribution']:
            analytics['category_distribution'][category] = 0
        analytics['category_distribution'][category] += 1
    
    def get_all_feedback(self, filters=None):
        feedbacks = st.session_state.feedback_database["feedbacks"]
        
        if filters:
            if filters.get('status'):
                feedbacks = [f for f in feedbacks if f['status'] == filters['status']]
            if filters.get('priority'):
                feedbacks = [f for f in feedbacks if f['priority'] == filters['priority']]
            if filters.get('category'):
                feedbacks = [f for f in feedbacks if f['category'] == filters['category']]
            if filters.get('date_range'):
                start_date, end_date = filters['date_range']
                feedbacks = [
                    f for f in feedbacks 
                    if start_date <= datetime.fromisoformat(f['timestamp']).date() <= end_date
                ]
        
        return sorted(feedbacks, key=lambda x: x['timestamp'], reverse=True)
    
    def update_feedback_status(self, feedback_id, new_status, admin_notes=None):
        for feedback in st.session_state.feedback_database["feedbacks"]:
            if feedback['id'] == feedback_id:
                feedback['status'] = new_status
                if admin_notes:
                    feedback['admin_notes'] = admin_notes
                feedback['last_updated'] = datetime.now().isoformat()
                return True
        return False
    
    def get_analytics(self):
        return st.session_state.feedback_database["analytics"]
    
    def export_feedback(self, format_type='csv'):
        feedbacks = self.get_all_feedback()
        df = pd.DataFrame(feedbacks)
        
        if format_type == 'csv':
            return df.to_csv(index=False)
        elif format_type == 'json':
            return df.to_json(orient='records', indent=2)
        else:
            return df.to_excel(index=False)

# Initialize feedback database
feedback_db = FeedbackDB()

# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "feedback_action" not in st.session_state:
    st.session_state.feedback_action = None
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "view_feedback" not in st.session_state:
    st.session_state.view_feedback = False
if "feedback_filters" not in st.session_state:
    st.session_state.feedback_filters = {}
if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="GenAI L1 Analyst Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply professional CSS
st.markdown("""
<style>
    /* Main background with professional color scheme */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    
    /* Headers with professional font */
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Chat bubbles */
    .chat-user {
        background-color: #e3f2fd;
        color: #1a237e;
        padding: 12px 16px;
        border-radius: 8px 8px 0 8px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #bbdefb;
    }
    
    .chat-assistant {
        background-color: #f1f8e9;
        color: #33691e;
        padding: 12px 16px;
        border-radius: 8px 8px 8px 0;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #dcedc8;
    }
    
    /* Buttons with professional style */
    .stButton button {
        border-radius: 4px;
        border: 1px solid #dee2e6;
        transition: all 0.2s ease;
        background-color: #ffffff;
        color: #495057;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .stButton button:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton button:focus {
        box-shadow: 0 0 0 3px rgba(0,123,255,0.25);
    }
    
    /* Primary buttons */
    .stButton button[kind="primary"] {
        background-color: #0d6efd;
        color: white;
        border-color: #0d6efd;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #0b5ed7;
        border-color: #0a58ca;
    }
    
    /* Secondary buttons */
    .stButton button[kind="secondary"] {
        background-color: #6c757d;
        color: white;
        border-color: #6c757d;
    }
    
    .stButton button[kind="secondary"]:hover {
        background-color: #5c636a;
        border-color: #565e64;
    }
    
    /* Feedback section */
    .feedback-section {
        background: #ffffff;
        padding: 20px;
        border-radius: 6px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metrics with professional styling */
    .metric-card {
        background: #ffffff;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card h3 {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 8px;
    }
    
    .metric-card h2 {
        font-size: 24px;
        color: #2c3e50;
        margin: 0;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
        border-radius: 4px;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #86b7fe !important;
        box-shadow: 0 0 0 0.25rem rgba(13,110,253,.25) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        color: #2c3e50;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-top: none;
        border-radius: 0 0 4px 4px;
    }
    
    /* Status indicators */
    .status-new {
        color: #0d6efd;
        font-weight: 500;
    }
    
    .status-reviewed {
        color: #198754;
    }
    
    .status-action {
        color: #fd7e14;
    }
    
    .status-closed {
        color: #6c757d;
    }
    
    /* Priority indicators */
    .priority-high {
        color: #dc3545;
        font-weight: 500;
    }
    
    .priority-medium {
        color: #fd7e14;
    }
    
    .priority-low {
        color: #198754;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Tab and section styling */
    .section-header {
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 10px;
        margin-bottom: 20px;
        color: #2c3e50;
    }
    
    /* Table styling */
    .dataframe {
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }
    
    .dataframe th {
        background-color: #f8f9fa;
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Alert styling */
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 4px;
        padding: 12px;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 4px;
        padding: 12px;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        padding: 12px;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        border-radius: 4px;
        padding: 12px;
    }
    
    /* Professional badge styling */
    .badge {
        background-color: #e9ecef;
        color: #495057;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid #dee2e6;
    }
    
    .badge-primary {
        background-color: #0d6efd;
        color: white;
    }
    
    .badge-success {
        background-color: #198754;
        color: white;
    }
    
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
    }
    
    .badge-danger {
        background-color: #dc3545;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Authentication (simple demo)
# -------------------------
def check_admin_access(password):
    # Simple demo authentication - replace with proper auth in production
    return password == "admin123"  # Change this in production

# -------------------------
# Sidebar for navigation
# -------------------------
with st.sidebar:
    st.markdown("<h2>Assistant Navigation</h2>", unsafe_allow_html=True)
    
    # User info section
    st.markdown("---")
    st.markdown("<h3>Analyst Information</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #dee2e6;'>
        <p style='margin: 0; color: #6c757d;'><strong>Role:</strong> L1 Support Analyst</p>
        <p style='margin: 0; color: #6c757d;'><strong>Status:</strong> <span style='color: #198754;'>Active</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Chat", use_container_width=True, key="chat_btn"):
            st.session_state.view_feedback = False
            st.rerun()
    
    with col2:
        if st.button("📊 Feedback", use_container_width=True, key="feedback_btn"):
            st.session_state.view_feedback = True
            st.rerun()
    
    st.markdown("---")
    
    # Quick stats
    analytics = feedback_db.get_analytics()
    st.markdown("<h3>Performance Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #dee2e6;'>
        <p style='margin: 0 0 8px 0;'><strong>Total Interactions:</strong> {len(st.session_state.messages)//2}</p>
        <p style='margin: 0 0 8px 0;'><strong>Feedback Provided:</strong> {analytics['total_feedback']}</p>
        <p style='margin: 0;'><strong>Satisfaction Rate:</strong> {(analytics['positive_count']/analytics['total_feedback']*100 if analytics['total_feedback'] > 0 else 0):.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Admin access
    st.markdown("<h3>Administrator Access</h3>", unsafe_allow_html=True)
    
    if not st.session_state.admin_mode:
        admin_pass = st.text_input("Admin Password", type="password", key="admin_pass")
        if st.button("Admin Login", use_container_width=True):
            if check_admin_access(admin_pass):
                st.session_state.admin_mode = True
                st.success("Admin mode activated")
                st.rerun()
            else:
                st.error("Invalid password")
    else:
        st.success("Admin Mode Active")
        if st.button("Logout Admin", use_container_width=True):
            st.session_state.admin_mode = False
            st.rerun()

# -------------------------
# Feedback View Page
# -------------------------
if st.session_state.view_feedback:
    st.markdown("<h1>Feedback Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Analytics Dashboard
    st.markdown("<h2 class='section-header'>Performance Metrics</h2>", unsafe_allow_html=True)
    
    analytics = feedback_db.get_analytics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Total Feedback</h3>
            <h2>{analytics['total_feedback']}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Positive</h3>
            <h2>{analytics['positive_count']}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Negative</h3>
            <h2>{analytics['negative_count']}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        satisfaction = (analytics['positive_count'] / analytics['total_feedback'] * 100) if analytics['total_feedback'] > 0 else 0
        satisfaction_color = "#198754" if satisfaction > 70 else ("#fd7e14" if satisfaction > 40 else "#dc3545")
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Satisfaction Rate</h3>
            <h2 style='color: {satisfaction_color};'>{satisfaction:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend chart
        if analytics['feedback_trend']:
            trend_df = pd.DataFrame(analytics['feedback_trend'])
            fig = px.line(trend_df, x='date', y=['positive', 'negative'], 
                         title='Feedback Trend (Last 7 Days)',
                         labels={'value': 'Count', 'variable': 'Type'},
                         color_discrete_map={'positive': '#198754', 'negative': '#dc3545'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#212529',
                title_font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category distribution
        if analytics['category_distribution']:
            cat_df = pd.DataFrame(list(analytics['category_distribution'].items()), 
                                 columns=['Category', 'Count'])
            fig = px.pie(cat_df, values='Count', names='Category', 
                        title='Feedback by Category',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#212529',
                title_font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feedback Management
    st.markdown("<h2 class='section-header'>Feedback Management</h2>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox("Status", 
                                   ["all", "new", "reviewed", "action_taken", "closed"])
    with col2:
        priority_filter = st.selectbox("Priority", 
                                     ["all", "high", "medium", "low"])
    with col3:
        category_filter = st.selectbox("Category", 
                                     ["all"] + list(set([f.get('category', 'uncategorized') 
                                                       for f in feedback_db.get_all_feedback()])))
    with col4:
        date_filter = st.date_input("Date Range", [])
    
    filters = {}
    if status_filter != "all":
        filters['status'] = status_filter
    if priority_filter != "all":
        filters['priority'] = priority_filter
    if category_filter != "all":
        filters['category'] = category_filter
    if len(date_filter) == 2:
        filters['date_range'] = date_filter
    
    filtered_feedback = feedback_db.get_all_feedback(filters)
    
    # Feedback list
    for feedback in filtered_feedback:
        status_class = f"status-{feedback['status']}"
        priority_class = f"priority-{feedback['priority']}"
        
        with st.expander(f"Feedback #{feedback['id']} - {feedback['timestamp'][:10]} - "
                        f"<span class='{status_class}'>{feedback['status']}</span> - "
                        f"<span class='{priority_class}'>{feedback['priority']}</span> - "
                        f"{feedback['category']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**User Interaction**")
                st.markdown(f"**Query:** {feedback['user_input']}")
                st.markdown(f"**Response:** {feedback['assistant_answer'][:200]}...")
                st.markdown(f"**Feedback Type:** {'Positive' if feedback['feedback_action'] == 'thumbs_up' else 'Negative'}")
                
                if feedback.get('feedback_text'):
                    st.markdown("**User Comments:**")
                    st.info(feedback['feedback_text'])
            
            with col2:
                st.markdown("**Management**")
                st.markdown(f"**Status:** <span class='{status_class}'>{feedback['status']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Priority:** <span class='{priority_class}'>{feedback['priority']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Category:** {feedback['category']}")
                st.markdown(f"**Received:** {feedback['timestamp']}")
                
                if st.session_state.admin_mode:
                    new_status = st.selectbox(
                        "Update Status",
                        ["new", "reviewed", "action_taken", "closed"],
                        index=["new", "reviewed", "action_taken", "closed"].index(feedback['status']),
                        key=f"status_{feedback['id']}"
                    )
                    
                    admin_notes = st.text_area(
                        "Admin Notes",
                        value=feedback.get('admin_notes', ''),
                        key=f"notes_{feedback['id']}"
                    )
                    
                    if st.button("Update", key=f"update_{feedback['id']}"):
                        if feedback_db.update_feedback_status(feedback['id'], new_status, admin_notes):
                            st.success("Feedback updated!")
                            st.rerun()
    
    # Export functionality
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Feedback Data**")
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        if st.button("Download Export", use_container_width=True):
            export_data = feedback_db.export_feedback(export_format.lower())
            
            if export_format == "CSV":
                st.download_button(
                    label="Download CSV",
                    data=export_data,
                    file_name="feedback_export.csv",
                    mime="text/csv",
                )
            elif export_format == "JSON":
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name="feedback_export.json",
                    mime="application/json",
                )
    
    with col2:
        if st.session_state.admin_mode:
            st.markdown("**Admin Tools**")
            if st.button("Refresh Analytics", use_container_width=True):
                st.rerun()
            if st.button("Clear All Feedback", use_container_width=True):
                if st.checkbox("Confirm deletion of ALL feedback data"):
                    st.session_state.feedback_database = {"feedbacks": [], "analytics": {"total_feedback": 0, "positive_count": 0, "negative_count": 0, "feedback_trend": [], "category_distribution": {}}}
                    st.success("All feedback data cleared")
                    st.rerun()

else:
    # -------------------------
    # Main Chat Interface
    # -------------------------
    st.markdown("""
    <h1 style='text-align:center'>GenAI L1 Analyst Assistant</h1>
    <p style='text-align:center; color:#6c757d; font-size:16px;'>
    Describe your IT issue. I'll guide you step-by-step through troubleshooting.
    </p>
    <hr style='border-color: #dee2e6;'>
    """, unsafe_allow_html=True)

    # Display past conversation
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        disclaimer_type = msg.get("disclaimer_type", None)
        
        if role == "user":
            st.markdown(f"<div class='chat-user'><b>You:</b> {content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'><b>Assistant:</b> {content}</div>", unsafe_allow_html=True)
            
            if disclaimer_type == "green":
                st.markdown("<div class='alert-success'>✅ Verified: This answer is grounded in KB.</div>", unsafe_allow_html=True)
            elif disclaimer_type == "yellow":
                st.markdown("<div class='alert-warning'>⚠️ Partially grounded in KB. Please validate.</div>", unsafe_allow_html=True)
            elif disclaimer_type == "red":
                st.markdown("<div class='alert-danger'>❌ Not found in KB. May not be reliable.</div>", unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Describe your issue or question...")
    if user_input:
        st.session_state.last_user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='chat-user'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
        
        with st.spinner("Searching for the best solution..."):
            result = query_rag(user_input)
            answer = result.get("answer", "I don't have an answer.")
            used_kb = result.get("used_kb", False)
            exact_match = result.get("exact_match", False)

            if exact_match and result.get("confidence", 0.0) >= 0.70:
                disclaimer_type = "green"
            elif used_kb and result.get("confidence", 0.0) >= 0.75:
                disclaimer_type = "green"
            elif used_kb and 0.55 <= result.get("confidence", 0.0) < 0.75:
                disclaimer_type = "yellow"
            else:
                disclaimer_type = "red"

            st.markdown(f"<div class='chat-assistant'><b>Assistant:</b> {answer}</div>", unsafe_allow_html=True)
            
            if disclaimer_type == "green":
                st.markdown("<div class='alert-success'>✅ Verified: This answer is grounded in KB.</div>", unsafe_allow_html=True)
            elif disclaimer_type == "yellow":
                st.markdown("<div class='alert-warning'>⚠️ Partially grounded in KB. Please validate.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-danger'>❌ Not found in KB. May not be reliable.</div>", unsafe_allow_html=True)

            st.session_state.last_answer = answer
            st.session_state.messages.append({"role": "assistant", "content": answer, "disclaimer_type": disclaimer_type})
            st.session_state.feedback_given = False
            st.session_state.feedback_action = None

    # Feedback form section
    if st.session_state.last_answer and not st.session_state.feedback_given:
        st.markdown("""
        <div class='feedback-section' style='margin-bottom:15px;'>
            <h4 style='color: black;text-align:center;'>Was this answer helpful?</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col2:
            if st.button("👍 Yes", use_container_width=True, type="primary"):
                st.session_state.feedback_action = "thumbs_up"
                
        with col3:
            if st.button("👎 No", use_container_width=True, type="secondary"):
                st.session_state.feedback_action = "thumbs_down"
                
        if st.session_state.feedback_action:
            with st.form(key="feedback_form"):
                feedback_text = st.text_area("Please share more details about your experience:")
                submit_feedback = st.form_submit_button("Submit Feedback", use_container_width=True)
                
                if submit_feedback:
                    feedback_data = {
                        "timestamp": datetime.now().isoformat(),
                        "user_input": st.session_state.last_user_input,
                        "assistant_answer": st.session_state.last_answer,
                        "feedback_action": st.session_state.feedback_action,
                        "feedback_text": feedback_text.strip()
                    }
                    feedback_id = feedback_db.store_feedback(feedback_data)
                    
                    st.success(f"Thank you for your feedback! (ID: #{feedback_id})")
                    st.session_state.feedback_given = True
                    st.rerun()

