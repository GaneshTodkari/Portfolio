# Home.py
import streamlit as st
import requests
import os
import time
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ganesh Todkari | Data & Business Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIG ---
API_URL = os.getenv("PORTFOLIO_API_URL", "http://127.0.0.1:8000")

# --- PATH SETUP ---
current_dir = Path(__file__).resolve().parent
resume_path = current_dir / "assets" / "resume.pdf"
profile_pic_path = current_dir / "assets" / "profile.png"

# --- POLISHED CSS (FONTS & THEME FORCING) ---
st.markdown("""
<style>
    /* Import Google Font for a modern look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    :root {
        --primary-blue: #1f77b4;
        --secondary-blue: #4a90e2;
        --dark-blue: #0f4c75;
        --light-blue: #e8f4fd;
        --text-dark: #2c3e50;
        --text-light: #7f8c8d;
        --background-white: #ffffff;
        --card-shadow: 0 4px 6px rgba(28, 83, 132, 0.1);
    }

    /* FORCE LIGHT THEME BACKGROUND */
    /* This ensures your design looks good even if user is in Dark Mode */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        color: var(--text-dark);
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e8ed;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, h4, .main-header, .section-header {
        font-family: 'Poppins', sans-serif;
    }

    /* COMPONENTS */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dark-blue);
        margin-bottom: 1rem;
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--dark-blue);
        margin: 2rem 0 1rem 0;
        padding-left: 0.8rem;
        border-left: 5px solid var(--primary-blue);
        background: linear-gradient(90deg, var(--light-blue), transparent);
    }
    
    .role-pill {
        background: var(--primary-blue);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.85rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: var(--background-white);
        padding: 1.5rem;
        border-radius: 12px;
        border-top: 4px solid var(--primary-blue);
        box-shadow: var(--card-shadow);
        margin: 0.5rem 0;
        text-align: center;
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(28, 83, 132, 0.15);
    }
    
    .project-card {
        background: var(--background-white);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        box-shadow: var(--card-shadow);
        height: 100%;
        transition: all 0.3s ease;
        border-top: 3px solid var(--primary-blue);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(28, 83, 132, 0.15);
        border-top: 3px solid var(--secondary-blue);
    }
    
    .skill-tag {
        background: var(--light-blue);
        color: var(--dark-blue);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #c5e1f6;
        font-weight: 600;
    }
    
    .info-box {
        background: var(--light-blue);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid var(--secondary-blue);
        margin: 1rem 0;
        color: var(--text-dark);
    }

    /* TABS Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        padding: 0 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-blue);
        color: white;
        border: none;
    }
    
    /* Make buttons fill width */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    if profile_pic_path.exists():
        col1, col2, col3 = st.columns([1,2,1])  # middle column wider
        with col1:
            st.image(str(profile_pic_path), width=220)
    else:
        st.markdown(
            """
            <div style='width: 200px; height: 140px; border-radius: 50%; background: var(--primary-blue);
                        display: flex; align-items: center; justify-content: center; color: white; font-size: 3rem; margin: 0 auto;'>
                GT
            </div>
            """,
            unsafe_allow_html=True
        )

    # Name and Title
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0; color: #1f77b4; font-size: 1.4rem;'>Ganesh Todkari</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center; color: #7f8c8d; margin-top: 5px; font-weight: 400; font-size: 1rem;'>MBA-IT @ SICSR</h3>",
        unsafe_allow_html=True
    )

    st.markdown("---")



# --- HERO SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="main-header">Transforming Data into Strategic Business Value</div>', unsafe_allow_html=True)

    # Updated Professional Story
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #0f4c75; margin-top: 0; font-size: 1.4rem;'>About Me</h3>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        My name is <strong>Ganesh Todkari</strong>, and I'm currently pursuing my MBA in IT at SICSR. 
        My journey began with a B.Sc. in Computer Science, where I built a strong foundation in how systems, data, and logic work.
        </p>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        A major turning point came when I moved from Tuljapur to Pune. 
        Being exposed to a larger tech ecosystem broadened my perspective — not through work experience, 
        but by finally seeing how technology supports business operations at scale. 
        This sparked a deeper curiosity in me: 
        <em>“Beyond writing code, what business problems are we really solving?”</em>
        </p>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        That question pushed me to move beyond pure programming and understand the <strong>business purpose</strong> 
        behind solutions. This is why I chose to pursue an MBA in IT.
        </p>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        During my MBA, I applied this mindset across multiple academic and freelance projects. 
        In the <strong>VybeRiders automation project</strong>, I collaborated with founders to gather requirements, 
        map processes, and help design an automated billing workflow. 
        Through <strong>Data Analytics</strong> and <strong>Machine Learning</strong> projects like 
        fraud detection, sales forecasting, and house-price prediction, 
        I transformed raw data into actionable insights using Python, SQL, and ML models.
        </p>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        These experiences helped me find my niche — a blend of <strong>Data Analysis</strong>, 
        <strong>Data Science</strong>, and <strong>Business Analysis</strong>. 
        I enjoy understanding problems end-to-end, analyzing data, and building solutions that make measurable business impact.
        </p>
        <p style='color: #2c3e50; line-height: 1.6; font-size: 1.05rem;'>
        I’m now excited to bring this dual perspective — technical foundation + business understanding — 
        to <strong>[Company Name]</strong> and contribute to meaningful, data-driven outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Core Competencies
    with st.container():
        st.markdown("### 🎯 Core Competencies")
        
        st.markdown("**Data Science & ML**")
        st.markdown("""
        <span class='skill-tag'>Python</span>
        <span class='skill-tag'>XGBoost</span>
        <span class='skill-tag'>Scikit-Learn</span>
        <span class='skill-tag'>K-Means</span>
        <span class='skill-tag'>Random Forest</span>
        <span class='skill-tag'>Regression</span>
        <span class='skill-tag'>Classification</span>
        <span class='skill-tag'>NLP</span>
        """, unsafe_allow_html=True)

        st.write("") # Spacer

        st.markdown("**Data Analytics & BI**")
        st.markdown("""
        <span class='skill-tag'>SQL</span>
        <span class='skill-tag'>Power BI</span>
        <span class='skill-tag'>Pandas</span>
        <span class='skill-tag'>NumPy</span>
        <span class='skill-tag'>Statistics</span>
        <span class='skill-tag'>EDA</span>
        <span class='skill-tag'>ETL</span>
        """, unsafe_allow_html=True)
    
        st.write("") # Spacer

        st.markdown("**Business Analysis**")
        st.markdown("""
        <span class='skill-tag'>Requirement Gathering</span>
        <span class='skill-tag'>Process Modeling</span>
        <span class='skill-tag'>Risk Analytics</span>
        <span class='skill-tag'>Inventory Planning</span>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        # Resume Download
        if resume_path.exists():
            with open(resume_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Master Resume",
                    data=pdf_file,
                    file_name="Ganesh_Todkari_Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
        # Contact Information
    st.markdown("### 📍 Contact Info")
    st.markdown(
        """
        <div style='background: white; padding: 1.2rem; border-radius: 8px; border: 1px solid #eee;'>
            <div style='margin-bottom: 10px;'>
                <strong>📧 Email</strong><br>
                <a href='mailto:ganesh697todkari@email.com' style='text-decoration: none; color: #1f77b4;'>ganesh697todkari@email.com</a>
            </div>
            <div style='margin-bottom: 10px;'>
                <strong>🔗 LinkedIn</strong><br>
                <a href='https://linkedin.com/in/GaneshTodkari' style='text-decoration: none; color: #1f77b4;'>linkedin.com/in/GaneshTodkari</a>
            </div>
            <div>
                <strong>💻 GitHub</strong><br>
                <a href='https://github.com/GaneshTodkari' style='text-decoration: none; color: #1f77b4;'>github.com/GaneshTodkari</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

# # --- AI ASSISTANT SECTION ---
# st.markdown("---")
# st.markdown('<div class="section-header">Interactive AI Assistant</div>', unsafe_allow_html=True)

# with st.container():
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         with st.container():
#             st.markdown("**Powered by Gemini AI** - Ask about my projects, technical skills, or business impact.")
            
#             if "messages" not in st.session_state:
#                 st.session_state.messages = [
#                     {"role": "assistant", "content": "Hello! I can explain Ganesh's technical skills, business impact, or project details. What would you like to know?"}
#                 ]

#             for msg in st.session_state.messages:
#                 with st.chat_message(msg["role"]):
#                     st.write(msg["content"])

#             prompt = st.chat_input("Ask about my experience...")
#             if prompt:
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 with st.chat_message("user"):
#                     st.write(prompt)

#                 with st.spinner("Analyzing..."):
#                     # SIMULATED BACKEND (Fall back if API is offline)
#                     try:
#                         resp = requests.post(f"{API_URL}/chat", json={"message": prompt}, timeout=3)
#                         if resp.status_code == 200:
#                             bot_reply = resp.json().get("response", "No response.")
#                         else:
#                             raise Exception("API Error")
#                     except:
#                         time.sleep(1) # Fake delay
#                         # Simple keyword matching for demo purposes
#                         p_lower = prompt.lower()
#                         if "fraud" in p_lower:
#                             bot_reply = "For the Fraud Detection project, I used SMOTE to handle imbalance and XGBoost for classification, achieving 0.99 AUC-ROC. It reduced false positives by 40%."
#                         elif "sales" in p_lower or "rossmann" in p_lower:
#                             bot_reply = "In the Rossmann Sales project, I predicted daily sales for 1,115 stores using XGBoost. I engineered features like 'Competition Distance' and 'Promo Intervals' to reach an R² of 0.85."
#                         elif "vybe" in p_lower or "rental" in p_lower:
#                             bot_reply = "For VybeRiders, I built a Flutter Admin Panel that automated billing. This reduced the manual checkout process from 10 minutes to just 15 seconds (97% efficiency gain)."
#                         else:
#                             bot_reply = "That's a great question! I have experience in Data Science (Python/ML), Analytics (SQL/Power BI), and Business Strategy. Check out my projects below for details."

#                 with st.chat_message("assistant"):
#                     st.write(bot_reply)
#                 st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    
#     with col2:
#         st.markdown("### 💡 Quick Questions")
#         quick_questions = [
#             "Explain the Rossmann sales model",
#             "Business impact of VybeRiders",
#             "Technical skills overview",
#             "Data visualization examples"
#         ]
#         for question in quick_questions:
#             if st.button(f"• {question}", use_container_width=True):
#                 st.session_state.messages.append({"role": "user", "content": question})
#                 # Trigger reload to show the question in chat (requires rerun)
#                 st.rerun()




# --- PROJECT PORTFOLIO SECTION ---
st.markdown("---")
st.markdown('<div class="section-header">Featured Projects</div>', unsafe_allow_html=True)

# Tabbed Project Layout
tab_ds, tab_da, tab_ba = st.tabs(["🧬 Data Science & ML", "📊 Data Analytics & BI", "💼 Business Analysis & Strategy"])

with tab_ds:
    col1, col2, col3 = st.columns(3)
    
    projects_ds = [
        {
            "title": "Retail Demand Forecasting",
            "subtitle": "Time Series Forecasting | XGBoost",
            "description": "Built ensemble models predicting daily sales for 1,115 stores, improving inventory planning accuracy by 25%.",
            "metrics": "R²: 0.85 | RMSE: 0.12",
            "skills": ["Time-Series Analysis", "XGBoost", "Cyclical Encoding", "Log Transformation", "Feature Engineering", "Pandas", "Statistical Analysis"],
            "link": "pages/1_📈_Rossmann_Sales.py"
        },
        {
            "title": "Credit Card Security Analysis",
            "subtitle": "Classification | Imbalanced Data",
            "description": "In financial security, the Needle in a Haystack problem is the norm. I worked with a dataset of over 10,000 transactions where only 1.5% (151 cases) were fraudulent.",
            "metrics": "AUC-ROC: 0.998",
            "skills": ["Scikit-learn", "SMOTE", "K-Means","Anomaly Detection", "XGBClassifier", "K-Means", "SQL", "Power BI", "Risk Analytics"],
            "link": "pages/2_🛡️_Fraud_Detection.py"
        },
        {
            "title": "Intelligent Real Estate Pricing Engine",
            "subtitle": "Regression Analysis",
            "description": "Real estate valuation engine using location clustering and physical features with 78% accuracy.",
            "metrics": "R²: 78%",
            "skills": ['Python', 'XGBoost', 'K-Means Clustering', 'Feature Engineering', 'RFECV', 'Predictive Modeling'],
            "link": "pages/5_🏠_House_Price.py"
        }
    ]
    
    for i, project in enumerate(projects_ds):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 500; font-size: 0.9rem;'>{project['subtitle']}</p>
                    <p style='color: #2c3e50; line-height: 1.5; font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #e8f4fd; padding: 0.7rem; border-radius: 6px; margin: 1rem 0;'>
                        <strong style='color: #0f4c75;'>Metrics:</strong> {project['metrics']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Explore Project →", key=f"btn_ds_{i}"):
                st.switch_page(project['link'])

with tab_da:
    col1, col2 = st.columns(2)
    
    projects_da = [
        {
            "title": "Fraud Patterns Dashboard",
            "subtitle": "Power BI | Data Visualization",
            "description": "Interactive dashboard analyzing $2M+ transactions, identifying fraud patterns and reducing investigation time by 60%.",
            "impact": "60% faster investigation",
            "skills": ["Power BI", "DAX", "SQL"],
            "link": "pages/3_📊_Fraud_Analysis.py"
        },
        {
            "title": "Contoso Retail Analytics",
            "subtitle": "SQL | Business Intelligence",
            "description": "Comprehensive retail analysis identifying top-performing products and regional sales trends leading to optimized inventory strategy.",
            "impact": "25% inventory cost reduction",
            "skills": ["SQL", "Power BI", "Excel"],
            "link": "pages/4_🏬_Contoso_Retail.py"
        }
    ]
    
    for i, project in enumerate(projects_da):
        with [col1, col2][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 500; font-size: 0.9rem;'>{project['subtitle']}</p>
                    <p style='color: #2c3e50; line-height: 1.5; font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #e8f4fd; padding: 0.7rem; border-radius: 6px; margin: 1rem 0;'>
                        <strong style='color: #0f4c75;'>Impact:</strong> {project['impact']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Dashboard →", key=f"btn_da_{i}"):
                st.switch_page(project['link'])

with tab_ba:
    col1, col2 = st.columns(2)
    
    projects_ba = [
        {
            "title": "VybeRiders Automation",
            "subtitle": "Business Process Improvement",
            "description": "Designed automated billing system replacing manual processes, reducing checkout time from 10 minutes to 15 seconds.",
            "impact": "97% process time reduction",
            "skills": ["Process Mapping", "Requirements", "Automation"]
        },
        
    ]
    
    for i, project in enumerate(projects_ba):
        with [col1, col2][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 500; font-size: 0.9rem;'>{project['subtitle']}</p>
                    <p style='color: #2c3e50; line-height: 1.5; font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #e8f4fd; padding: 0.7rem; border-radius: 6px; margin: 1rem 0;'>
                        <strong style='color: #0f4c75;'>Impact:</strong> {project['impact']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TECHNICAL EXPERTISE SECTION ---
st.markdown("---")
st.markdown('<div class="section-header">Technical Expertise</div>', unsafe_allow_html=True)

skill_col1, skill_col2, skill_col3 = st.columns(3)

with skill_col1:
    with st.container():
        st.markdown("#### 🐍 Data Science & ML")
        st.markdown("""
        <div class='info-box'>
            <ul style='padding-left: 20px; margin: 0;'>
                <li><strong>Algorithms:</strong>Linear/Logistic Regression, Random Forest, XGBoost (Regressor & Classifier), Decision Trees.</li>
                <li><strong>Statistical Analysis:</strong> Hypothesis Testing, A/B Testing</li>
                <li><strong>Feature Engineering:</strong> Cyclical Time-Series (Sine/Cosine), Log-Transformations</li>
                <li><strong>GenAI:</strong> NLP, Prompt Engineering, LLM Utilization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with skill_col2:
    with st.container():
        st.markdown("#### 📊 Data Analytics & BI")
        st.markdown("""
        <div class='info-box'>
            <ul style='padding-left: 20px; margin: 0;'>
                <li><strong>BI Tools:</strong> Power BI, Tableau, Excel</li>
                <li><strong>Database:</strong> SQL, NoSQL</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy, ETL</li>
                <li><strong>Statistical Analysis:</strong> Hypothesis Testing, VIF (Multicollinearity checks), Outlier Detection (IQR & Percentile Capping), Correlation Analysis (Pearson/Spearman).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with skill_col3:
    with st.container():
        st.markdown("#### 💼 Business Analysis")
        st.markdown("""
        <div class='info-box'>
            <ul style='padding-left: 20px; margin: 0;'>
                <li><strong>Core Skills:</strong> Requirement Gathering, Process Modeling, Stakeholder Management.</li>
                <li><strong>Process:</strong> Workflow Analysis, BPMN</li>
                <li><strong>Tools:</strong> JIRA</li>
                <li><strong>Agile:</strong> Scrum, Sprint Planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <h4 style='color: #0f4c75; margin-bottom: 1rem;'>Open to New Opportunities</h4>
    <p style='margin-bottom: 1.5rem;'>Available for full-time positions in Data Science, Business Analytics, and Data Analysis roles</p>
    <div style='margin: 1rem 0;'>
        <a href='mailto:ganesh697todkari@email.com' style='margin: 0 1rem; color: #1f77b4; text-decoration: none;'>📧 Email Me</a>
        <a href='https://linkedin.com/in/GaneshTodkari' style='margin: 0 1rem; color: #1f77b4; text-decoration: none;'>🔗 LinkedIn</a>
        <a href='https://github.com/GaneshTodkari' style='margin: 0 1rem; color: #1f77b4; text-decoration: none;'>💻 GitHub</a>
    </div>
    <p style='margin-top: 1.5rem; font-size: 0.8rem;'>Designed & Built with Streamlit & Python • Ganesh Todkari</p>
</div>
""", unsafe_allow_html=True)