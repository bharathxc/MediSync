"""
MediSync AI — Unified Streamlit Application
Dual Login: Member & Employee Access
"""
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from config import (
    LLM_BACKEND,
    OLLAMA_MODEL,
    GOOGLE_MODEL,
    CRM_DATA_PATH,
)
from agents.agent import MediSyncAgent
from nlp.hybrid_pipeline import HybridDeidentifier
from nlp.bias_auditor import BiasAuditor

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediSync AI | Healthcare Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .login-container {
        max-width: 420px;
        margin: 3rem auto;
        padding: 2.5rem;
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-header h1 {
        color: #ff8c57 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .login-header p {
        color: #7f8fa6 !important;
        font-size: 0.9rem !important;
    }
    .login-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stPassword > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #121e36 50%, #1a2744 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #ff6b2b 0%, #e8451e 50%, #c73018 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(255, 107, 43, 0.25);
    }
    .employee-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2240 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.25);
    }
    .member-card {
        background: linear-gradient(135deg, rgba(255,107,43,0.12) 0%, rgba(30,60,114,0.15) 100%);
        border: 1px solid rgba(255,107,43,0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .member-card h4 { color: #ff8c57 !important; margin: 0 0 0.5rem 0 !important; }
    .member-card .detail { color: #a0b4cc !important; font-size: 0.82rem; margin: 0.2rem 0; }
    .plan-badge {
        display: inline-block; padding: 0.2rem 0.75rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    }
    .plan-basic { background: rgba(52, 152, 219, 0.2); color: #5dade2; }
    .plan-premium { background: rgba(155, 89, 182, 0.2); color: #c39bd3; }
    .status-active { background: rgba(39, 174, 96, 0.2); color: #58d68d; }
    .employee-badge {
        display: inline-block; padding: 0.2rem 0.75rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
        background: rgba(52, 152, 219, 0.2); color: #5dade2;
    }
    .footer { text-align: center; padding: 1rem; color: rgba(255,255,255,0.3); font-size: 0.75rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

backend_name = OLLAMA_MODEL if LLM_BACKEND == "ollama" else GOOGLE_MODEL

# ─── Employee Credentials ──────────────────────────────────────────
EMPLOYEE_CREDENTIALS = {
    "admin@optum.com": {"password": "admin123", "name": "Sarah Mitchell", "role": "HR Administrator"},
    "provider@optum.com": {"password": "provider123", "name": "Dr. James Chen", "role": "Healthcare Provider"},
    "analyst@optum.com": {"password": "analyst123", "name": "Mike Johnson", "role": "Data Analyst"},
}

# ─── Load CRM Data ───────────────────────────────────────────────────────────
@st.cache_data
def load_crm_data():
    with open(CRM_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data["members"]

# ─── Session State ────────────────────────────────────────────────────────────
if "auth_type" not in st.session_state:
    st.session_state.auth_type = None
if "employee_info" not in st.session_state:
    st.session_state.employee_info = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "selected_member" not in st.session_state:
    st.session_state.selected_member = None

def initialize_agent():
    agent = MediSyncAgent()
    if st.session_state.selected_member:
        agent.set_member(st.session_state.selected_member)
    return agent

def logout():
    st.session_state.auth_type = None
    st.session_state.employee_info = None
    st.session_state.messages = []
    st.session_state.agent = None
    st.session_state.selected_member = None
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.auth_type is None:
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <div class="login-icon">🏥</div>
            <h1>MediSync AI</h1>
            <p>Unified Healthcare Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Member Login", key="btn_member", use_container_width=True):
            st.session_state.auth_type = "member"
            st.rerun()
    with col2:
        if st.button("👨‍💼 Employee Login", key="btn_employee", use_container_width=True):
            st.session_state.auth_type = "employee"
            st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ Portfolio Project — Not for clinical use")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# EMPLOYEE LOGIN FORM
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.auth_type == "employee" and st.session_state.employee_info is None:
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <div class="login-icon">👨‍💼</div>
            <h1>Employee Portal</h1>
            <p>Optum HR & Healthcare Provider Access</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("employee_login"):
        email = st.text_input("Email", placeholder="admin@optum.com")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Sign In", use_container_width=True)
        
        if submit:
            if email in EMPLOYEE_CREDENTIALS:
                if EMPLOYEE_CREDENTIALS[email]["password"] == password:
                    st.session_state.employee_info = {
                        "email": email,
                        "name": EMPLOYEE_CREDENTIALS[email]["name"],
                        "role": EMPLOYEE_CREDENTIALS[email]["role"]
                    }
                    st.rerun()
                else:
                    st.error("Incorrect password")
            else:
                st.error("Invalid email. Try: admin@optum.com / admin123")
    
    if st.button("← Back to Login"):
        st.session_state.auth_type = None
        st.rerun()
    
    st.markdown("---")
    st.caption("Demo: admin@optum.com / admin123")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
else:
    # ─── SIDEBAR (Unified for all authenticated views) ───────────────────
    with st.sidebar:
        if st.session_state.auth_type == "member":
            st.markdown("""
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <div style="font-size: 2.5rem;">🏥</div>
                <h2 style="color: #ff8c57; margin: 0;">MediSync AI</h2>
                <p style="color: #5d7a99; font-size: 0.8rem; margin: 0;">Member Portal</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            # Member Selection
            st.markdown("#### 👤 Select Member")
            members = load_crm_data()
            member_names = [f"{m['name']} ({m['plan_type']})" for m in members]
            member_options = ["-- Select --"] + member_names
            selected_name = st.selectbox("Member:", member_options, key="member_select")
            
            if selected_name != "-- Select --":
                # Find the selected member
                for m in members:
                    if f"{m['name']} ({m['plan_type']})" == selected_name:
                        if st.session_state.selected_member != m:
                            st.session_state.selected_member = m
                            st.session_state.agent = None
                            st.session_state.messages = []
                        break
            else:
                st.session_state.selected_member = None
            
            # Show member card if selected
            if st.session_state.selected_member:
                member = st.session_state.selected_member
                plan_class = "plan-premium" if member['plan_type'] == 'Premium' else "plan-basic"
                status_class = "status-active" if member['status'] == 'Active' else "status-inactive"
                st.markdown(f"""
                <div class="member-card">
                    <h4>{member['name']}</h4>
                    <div class="detail">
                        <span class="{plan_class} plan-badge">{member['plan_type']}</span>
                        <span class="status-active" style="display:inline-block;padding:0.15rem 0.6rem;border-radius:20px;font-size:0.7rem;font-weight:600;">{member['status']}</span>
                    </div>
                    <div class="detail"><strong>ID:</strong> {member['member_id']}</div>
                    <div class="detail"><strong>PCP:</strong> {member['pcp']}</div>
                    <div class="detail"><strong>Group:</strong> {member['group_number']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Employee view
            emp = st.session_state.employee_info
            st.markdown(f"""
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <div style="font-size: 2.5rem;">🏢</div>
                <h2 style="color: #5dade2; margin: 0;">MediSync AI</h2>
                <p style="color: #5d7a99; font-size: 0.8rem; margin: 0;">Employee Portal</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(52,152,219,0.1); border: 1px solid rgba(52,152,219,0.3);
                        border-radius: 8px; padding: 0.75rem;">
                <div style="color: #5dade2; font-weight: 600;">👤 {emp['name']}</div>
                <div style="color: #7f8fa6; font-size: 0.75rem;">{emp['role']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="background: rgba(39,174,96,0.1); border: 1px solid rgba(39,174,96,0.3);
                    border-radius: 8px; padding: 0.6rem; margin-bottom: 0.5rem;">
            <div style="color: #58d68d; font-size: 0.8rem; font-weight: 600;">{LLM_BACKEND.upper()} ({backend_name})</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Log Out", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        if st.session_state.auth_type == "member":
            st.markdown("#### 🛡️ Safety")
            st.caption("✅ PII Masking • ✅ RAG Only")
        else:
            st.markdown("#### 🛡️ Admin")
            st.caption("✅ De-ID • ✅ Bias Audit")

    # ─── MAIN CONTENT ───────────────────────────────────────────────────
    if st.session_state.auth_type == "member":
        st.markdown("""
        <div class="main-header">
            <h1>🏥 MediSync AI</h1>
            <p>Your Intelligent Healthcare Benefits Navigator</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome message
        if not st.session_state.messages:
            if st.session_state.selected_member:
                welcome = f"Hello, **{st.session_state.selected_member['name']}**! 👋 I'm MediSync AI on your **{st.session_state.selected_member['plan_type']} Plan**."
            else:
                welcome = "Welcome to **MediSync AI**! 👋 Select a member from the sidebar to get started."
            st.session_state.messages.append({"role": "assistant", "content": welcome})
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about your health plan benefits..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            if st.session_state.agent is None:
                st.session_state.agent = initialize_agent()
            
            with st.chat_message("assistant", avatar="🏥"):
                with st.spinner("Analyzing..."):
                    try:
                        response = st.session_state.agent.chat(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    else:
        emp = st.session_state.employee_info
        st.markdown(f"""
        <div class="employee-header">
            <h1>🏢 MediSync AI — Employee Portal</h1>
            <p>Welcome, {emp['name']} | {emp['role']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🏥 Member Support", "🛡️ Clinical NLP", "📊 Analytics"])
        
        with tab1:
            st.markdown("### Member Support Chat")
            
            with st.expander("👤 Member Lookup", expanded=True):
                members = load_crm_data()
                member_names = [f"{m['name']} ({m['member_id']}) - {m['plan_type']}" for m in members]
                selected = st.selectbox("Select Member:", range(len(members)),
                    format_func=lambda i: member_names[i])
                
                if selected is not None:
                    member = members[selected]
                    with st.container():
                        plan_cls = "plan-premium" if member['plan_type']=='Premium' else "plan-basic"
                        st.markdown(f"""
                        <div class="member-card">
                            <h4>{member['name']}</h4>
                            <span class="{plan_cls} plan-badge">{member['plan_type']}</span>
                            <div class="detail"><strong>ID:</strong> {member['member_id']}</div>
                            <div class="detail"><strong>DOB:</strong> {member['date_of_birth']}</div>
                            <div class="detail"><strong>PCP:</strong> {member['pcp']}</div>
                            <div class="detail"><strong>Status:</strong> {member['status']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": "👋 Hello! How can I help you assist members today?"})
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
            
            if prompt := st.chat_input("Ask about member benefits..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                if st.session_state.agent is None:
                    st.session_state.agent = initialize_agent()
                
                with st.chat_message("assistant", avatar="🏥"):
                    with st.spinner("Analyzing..."):
                        try:
                            response = st.session_state.agent.chat(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        with tab2:
            st.markdown("### 🛡️ Clinical NLP Safeguard")
            st.caption(f"Powered by Presidio + {backend_name}")
            
            @st.cache_resource
            def load_pipeline():
                return HybridDeidentifier()
            
            @st.cache_resource
            def load_auditor():
                return BiasAuditor()
            
            try:
                pipeline = load_pipeline()
                auditor = load_auditor()
            except Exception as e:
                st.error(f"Error initializing: {str(e)}")
                st.stop()
            
            subtab1, subtab2 = st.tabs(["🩺 De-Identification", "⚖️ Bias Audit"])
            
            with subtab1:
                default_text = "Patient John Smith was admitted to Mayo Clinic on 2024-05-12. Patient has Down's syndrome. Diagnosed with Parkinson's disease. Call 555-123-4567."
                input_text = st.text_area("Clinical Note:", value=default_text, height=120)
                
                if st.button("Run De-Identification", type="primary"):
                    with st.spinner("Processing..."):
                        result = pipeline.process_note(input_text)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("1. Traditional NER")
                            st.info(result['traditional_redaction'])
                        with col2:
                            st.subheader("2. GenAI Refined")
                            st.success(result['hybrid_redaction'])
            
            with subtab2:
                if st.button("Run Bias Audit"):
                    with st.spinner("Evaluating..."):
                        df, summary = auditor.evaluate_model()
                        fig = px.bar(summary, x='demographic_group', y='detection_rate',
                            color='demographic_group', title="Detection Rate by Demographic",
                            text_auto='.1%')
                        fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
                        fig.add_hline(y=1.0, line_dash="dash", line_color="green")
                        st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("### 📊 Analytics Dashboard")
            
            members = load_crm_data()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Members", len(members))
            with col2:
                premium = sum(1 for m in members if m['plan_type'] == 'Premium')
                st.metric("Premium", premium)
            with col3:
                basic = sum(1 for m in members if m['plan_type'] == 'Basic')
                st.metric("Basic", basic)
            with col4:
                active = sum(1 for m in members if m['status'] == 'Active')
                st.metric("Active", active)
            
            st.markdown("---")
            
            plan_counts = pd.DataFrame([{"Plan": m["plan_type"], "Status": m["status"]} for m in members])
            col_a, col_b = st.columns(2)
            with col_a:
                fig1 = px.pie(plan_counts, names='Plan', title="Plan Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig1, use_container_width=True)
            with col_b:
                fig2 = px.histogram(plan_counts, x='Status', color='Plan', 
                    title="Status by Plan", barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig2, use_container_width=True)