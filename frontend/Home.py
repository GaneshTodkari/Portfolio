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

# --- PATH SETUP ---
# Ensure these paths exist in your project folder
current_dir = Path(__file__).resolve().parent
resume_path = current_dir / "assets" / "resume.pdf"
profile_pic_path = current_dir / "assets" / "profile.png"

# --- BACKEND URL (from Streamlit secrets) ---
# Make sure you set this in Streamlit Cloud: BACKEND_URL = "https://portfolio-i8re.onrender.com"
BACKEND_URL = st.secrets.get("BACKEND_URL", "https://portfolio-i8re.onrender.com")

# --- POLISHED CSS (FONTS & THEME FORCING) ---
st.markdown("""
<style>
    /* Import Google Font for a modern look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-blue: #1f77b4;
        --secondary-blue: #4a90e2;
        --dark-blue: #0f4c75;
        /* --light-blue: #f0f7fd; Removed as it's no longer used in heavy boxes */
        --text-dark: #2c3e50;
        --text-medium: #546e7a; /* A nice readable dark gray for secondary text */
        --background-white: #ffffff;
        --card-shadow: 0 2px 8px rgba(0,0,0,0.05);
        --hover-shadow: 0 8px 16px rgba(31, 119, 180, 0.15);
    }

    /* BASE TYPOGRAPHY & THEME FORCING (Keeps light mode consistent) */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        color: var(--text-dark);
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e8ed;
    }

    /* --- FIX 1: IMPROVE SIDEBAR NAV CONTRAST --- */
    /* Target inactive sidebar links to make them darker and readable */
    [data-testid="stSidebarNav"] a {
        color: var(--text-medium) !important;
        font-weight: 500;
    }
    /* Ensure active link is highlighted correctly */
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        color: var(--primary-blue) !important;
        font-weight: 700;
    }

    h1, h2, h3, h4, .main-header, .section-header {
        font-family: 'Poppins', sans-serif;
        color: var(--dark-blue);
    }
    
    p {
        line-height: 1.7;
        color: var(--text-medium);
    }

    /* COMPONENTS */
    
    /* --- FIX 2: CLEANER HERO HEADER DESIGN --- */
    /* Removed heavy background box and borders for a typographic approach */
    .hero-header-container {
        padding: 1rem 0 2rem 0;
        margin-bottom: 1rem;
    }

    .main-header {
        font-size: 2.5rem; /* Increased size for impact */
        font-weight: 800; /* Heavier weight */
        color: var(--dark-blue);
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e1e8ed;
    }
    
    /* Force rounded sidebar image */
    [data-testid="stSidebar"] img {
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid var(--primary-blue);
        padding: 3px;
    }

    /* Project Cards */
    .project-card {
        background: var(--background-white);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #edf2f7;
        box-shadow: var(--card-shadow);
        height: 100%;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--hover-shadow);
        border-color: var(--secondary-blue);
    }
    
    /* Skill Tags */
    .skill-tag {
        background: #ffffff;
        color: var(--primary-blue);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 0.25rem;
        display: inline-block;
        border: 1px solid var(--secondary-blue);
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .info-box {
        background: var(--background-white);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
    }

    /* TABS Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: var(--background-white);
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0 20px;
        font-weight: 600;
        color: var(--text-medium);
        transition: all 0.2s;
    }
    
    /* --- FIX 3: ACTIVE TAB CONTRAST --- */
    /* Force text and icons inside the active tab to be white */
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] svg {
        background-color: var(--primary-blue) !important;
        color: #ffffff !important;
        fill: #ffffff !important; /* For SVG icons if present */
        border: none;
        box-shadow: 0 4px 6px rgba(31, 119, 180, 0.2);
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.1s;
    }
    .stButton button:active {
         transform: scale(0.98);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.write("") # Top spacing
    if profile_pic_path.exists():
        # Using columns to center the image perfectly
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        with col2:
            st.image(str(profile_pic_path), width=180)
    else:
        # Fallback placeholder
        st.markdown(
            """
            <div style='width: 160px; height: 160px; border-radius: 50%; background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
                        display: flex; align-items: center; justify-content: center; color: white; font-size: 3rem; margin: 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
                GT
            </div>
            """,
            unsafe_allow_html=True
        )

    # Name and Title
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 5px; margin-top: 20px; color: #0f4c75; font-size: 1.6rem; font-weight: 700;'>Ganesh Todkari</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center; color: #546e7a; margin-top: 0; font-weight: 500; font-size: 1rem;'>MBA-IT @ SICSR</h3>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    # Contact Info in Sidebar
    st.markdown("### 📍 Contact")
    st.markdown(
        """
        <div style='font-size: 0.9rem;'>
            <div style='margin-bottom: 8px;'>
                <a href='mailto:ganesh697todkari@email.com' style='text-decoration: none; color: #546e7a;'>📧 Email Me</a>
            </div>
            <div style='margin-bottom: 8px;'>
                <a href='https://linkedin.com/in/GaneshTodkari' style='text-decoration: none; color: #546e7a;'>🔗 LinkedIn Profile</a>
            </div>
            <div>
                <a href='https://github.com/GaneshTodkari' style='text-decoration: none; color: #546e7a;'>💻 GitHub Portfolio</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- MAIN CONTENT START ---

# --- HERO SECTION ---
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("""
    <div class="hero-header-container">
        <div class="main-header">Turning Data into Strategy</div>
        <p style="font-size: 1.25rem; color: var(--text-medium); margin-bottom: 0; font-weight: 400;">
            Bridging the gap between complex datasets and actionable business strategy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Professional Story
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: var(--dark-blue); margin-top: 0; font-size: 1.4rem; display: flex; align-items: center; gap: 10px;'>
            👋 About Me
        </h3>
        <p>
        Hi, I’m <strong>Ganesh Todkari</strong>. I come from the culturally rich town of Tuljapur, where I completed my early schooling. My fascination with how things work led me to pursue a <strong>B.Sc in Computer Science</strong>, giving me a strong foundation in technical systems.
        </p>
        <p>
        However, during my studies, I found myself asking a critical question: <em>'How does this technology actually help businesses achieve their goals?'</em>
        </p>
        <p>
        That curiosity drove me to pursue an <strong>MBA in IT</strong> to nurture my managerial skills and bridge the gap between raw code and business strategy. I’ve turned this curiosity into action through hands-on projects:
        </p>
        <ul style="color: var(--text-medium); line-height: 1.7; margin-bottom: 1rem;">
            <li><strong>Business Analysis:</strong> Designed automated billing workflows for <strong>VybeRiders</strong>, bridging the gap between operations and tech.</li>
            <li><strong>Data Science:</strong> Built <strong>Machine Learning models</strong> for Fraud Detection and Sales Forecasting using <strong>Python & SQL</strong>.</li>
        </ul>
        <p>
        These experiences have defined my niche at the intersection of <strong>Data Science</strong> and <strong>Business Analysis</strong>.
        </p>
        <p style='margin-bottom: 0; font-size: 0.95rem; color: var(--text-medium); border-top: 1px solid #eee; padding-top: 1rem;'>
        <strong>Offline:</strong> When I’m not working, you can find me reading Manga, watching Anime, or cheering for my favorite cricket team.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Core Competencies
    with st.container():
        st.markdown("### 🎯 Core Competencies")
        
        st.markdown("**🧠 Data Science & ML**")
        st.markdown("""
        <span class='skill-tag'>Python</span>
        <span class='skill-tag'>XGBoost</span>
        <span class='skill-tag'>Scikit-Learn</span>
        <span class='skill-tag'>Random Forest</span>
        <span class='skill-tag'>Regression & Classification</span>
        <span class='skill-tag'>NLP & GenAI</span>
        """, unsafe_allow_html=True)

        st.write("") # Spacer

        st.markdown("**📊 Data Analytics & BI**")
        st.markdown("""
        <span class='skill-tag'>SQL</span>
        <span class='skill-tag'>Power BI</span>
        <span class='skill-tag'>Pandas & NumPy</span>
        <span class='skill-tag'>Statistical Analysis</span>
        <span class='skill-tag'>EDA</span>
        <span class='skill-tag'>ETL Pipelines</span>
        """, unsafe_allow_html=True)
    
        st.write("") # Spacer

        st.markdown("**💼 Business Analysis**")
        st.markdown("""
        <span class='skill-tag'>Requirement Gathering</span>
        <span class='skill-tag'>Process Modeling (BPMN)</span>
        <span class='skill-tag'>Risk Analytics</span>
        <span class='skill-tag'>Stakeholder Management</span>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        st.write("") # Spacer
        # Resume Download Button
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



# --- PROJECT PORTFOLIO SECTION ---
st.write("")
st.markdown('<div class="section-header">Featured Projects</div>', unsafe_allow_html=True)

# Tabbed Project Layout
tab_ds, tab_da, tab_ba = st.tabs(["🧬 Data Science & ML", "📊 Data Analytics & BI", "💼 Business Analysis & Strategy"])

with tab_ds:
    col1, col2, col3 = st.columns(3, gap="medium")
    
    projects_ds = [
        {
            "title": "Retail Demand Forecasting",
            "subtitle": "XGBoost | Time Series",
            "description": "Built ensemble models predicting daily sales for 1,115 stores, directly improving inventory planning accuracy.",
            "metrics": "R²: 0.85 | Reduced Forecast Error by 15%",
            "skills": ["Python", "XGBoost", "Time-Series", "Feature Engineering"],
            "link": "pages/1_📈_Retail_Demand_Forecasting.py"
        },
        {
            "title": "Credit Card Fraud Detection",
            "subtitle": "Anomaly Detection | Imbalanced Data",
            "description": "Solved the 'Needle in a Haystack' problem in financial data, identifying rare fraudulent transactions with high precision.",
            "metrics": "AUC-ROC: 0.998",
            "skills": ["Scikit-learn", "SMOTE", "XGBClassifier", "Risk Analytics"],
            "link": "pages/2_🛡️_Credit_Card_Security_Analysis.py"
        },
        {
            "title": "Real Estate Pricing Engine",
            "subtitle": "Regression Analysis",
            "description": "Developed a valuation engine using location clustering and physical features to predict property prices accurately.",
            "metrics": "Accuracy: 78%",
            "skills": ['Python', 'K-Means Clustering', 'Predictive Modeling', 'EDA'],
            "link": "pages/5_🏠_House_Price.py"
        }
    ]
    
    for i, project in enumerate(projects_ds):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0; margin-bottom: 5px;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 600; font-size: 0.85rem; margin-bottom: 10px;'>{project['subtitle']}</p>
                    <p style='font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #f0f7fd; padding: 0.5rem 0.8rem; border-radius: 6px; margin: 1rem 0; font-size: 0.9rem;'>
                        <strong style='color: #0f4c75;'>Key Metric:</strong> {project['metrics']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button("View Case Study", key=f"btn_ds_{i}"):
                st.switch_page(project['link'])

with tab_da:
    col1, col2 = st.columns(2, gap="medium")
    
    projects_da = [
        {
            "title": "Insurance Fraud Dashboard",
            "subtitle": "Power BI | Data Visualization",
            "description": "Interactive dashboard analyzing $2M+ transactions to identify patterns, drastically reducing manual investigation time.",
            "impact": "60% faster investigation workflows",
            "skills": ["Power BI", "DAX", "SQL", "Data Modeling"],
            "link": "pages/3_📊_Insurance_Fraud_Claim_Analysis.py"
        },
        {
            "title": "Retail Sales & Inventory Analytics",
            "subtitle": "SQL | Business Intelligence",
            "description": "Comprehensive analysis identifying top products and regional trends leading to an optimized inventory strategy.",
            "impact": "Identified 25% potential inventory cost reduction",
            "skills": ["SQL", "Power BI", "Excel", "Strategic Analysis"],
            "link": "pages/4_🏬_Contoso_Retail.py"
        }
    ]
    
    for i, project in enumerate(projects_da):
        with [col1, col2][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0; margin-bottom: 5px;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 600; font-size: 0.85rem; margin-bottom: 10px;'>{project['subtitle']}</p>
                    <p style='font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #f0f7fd; padding: 0.5rem 0.8rem; border-radius: 6px; margin: 1rem 0; font-size: 0.9rem;'>
                        <strong style='color: #0f4c75;'>Business Impact:</strong> {project['impact']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button("View Dashboard", key=f"btn_da_{i}"):
                st.switch_page(project['link'])

with tab_ba:
    col1, col2 = st.columns(2, gap="medium")
    
    projects_ba = [
        {
            "title": "VybeRiders Process Automation",
            "subtitle": "BPMN | Automation",
            "description": "Mapped existing workflows and designed an automated billing system to replace manual checkout processes.",
            "impact": "Reduced checkout time from 10 mins to 15 seconds (97% gain)",
            "skills": ["Process Mapping", "Requirement Gathering", "Automation Design"],
            # Assuming you might have a page for this, otherwise remove button
             "link": None 
        },
        
    ]
    
    for i, project in enumerate(projects_ba):
        with [col1, col2][i]:
            st.markdown(f"""
            <div class='project-card'>
                <div>
                    <h4 style='color: #0f4c75; margin-top: 0; margin-bottom: 5px;'>{project['title']}</h4>
                    <p style='color: #1f77b4; font-weight: 600; font-size: 0.85rem; margin-bottom: 10px;'>{project['subtitle']}</p>
                    <p style='font-size: 0.95rem;'>{project['description']}</p>
                    <div style='background: #f0f7fd; padding: 0.5rem 0.8rem; border-radius: 6px; margin: 1rem 0; font-size: 0.9rem;'>
                        <strong style='color: #0f4c75;'>Business Impact:</strong> {project['impact']}
                    </div>
                </div>
                <div>
                    {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if project.get('link'):
                 if st.button("View Details", key=f"btn_ba_{i}"):
                    st.switch_page(project['link'])
            else:
                 st.button("Details Coming Soon", key=f"btn_ba_{i}", disabled=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <h4 style='color: #0f4c75; margin-bottom: 0.5rem;'>Let's Connect</h4>
    <p style='margin-bottom: 1.5rem; font-size: 0.9rem;'>Open to full-time opportunities in Data Science & Analytics.</p>
    <div style='margin: 1rem 0; font-size: 1.1rem;'>
        <a href='mailto:ganesh697todkari@email.com' style='margin: 0 1.5rem; color: #1f77b4; text-decoration: none; font-weight: 600;'>📧 Email</a>
        <a href='https://linkedin.com/in/GaneshTodkari' style='margin: 0 1.5rem; color: #1f77b4; text-decoration: none; font-weight: 600;'>🔗 LinkedIn</a>
        <a href='https://github.com/GaneshTodkari' style='margin: 0 1.5rem; color: #1f77b4; text-decoration: none; font-weight: 600;'>💻 GitHub</a>
    </div>
    <p style='margin-top: 2rem; font-size: 0.75rem; color: #b0bec5;'>© 2023 Ganesh Todkari • Built with Python & Streamlit</p>
</div>
""", unsafe_allow_html=True)