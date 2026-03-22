import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Ganesh Todkari | Data, Analytics, and Business Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_dir = Path(__file__).resolve().parent
resume_path = current_dir / "assets" / "resume.pdf"
profile_pic_path = current_dir / "assets" / "profile.png"
vyberiders_dir = current_dir / "assets" / "vyberiders"


def project_card(project: dict, key: str):
    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-topline">{project['category']}</div>
            <h3 class="project-title">{project['title']}</h3>
            <div class="project-row">
                <span class="project-label">Problem</span>
                <p>{project['problem']}</p>
            </div>
            <div class="project-row">
                <span class="project-label">Solution</span>
                <p>{project['solution']}</p>
            </div>
            <div class="project-result">
                <span class="project-label">Result</span>
                <div>{project['result']}</div>
            </div>
            <div class="tag-row">
                {"".join([f"<span class='tag'>{skill}</span>" for skill in project["skills"]])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if project.get("link"):
        if st.button(project["button"], key=key):
            st.switch_page(project["link"])
    else:
        st.button("Detailed Case Study Coming Soon", key=key, disabled=True)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --accent: #0c7b93;
        --accent-strong: #145da0;
        --accent-soft: rgba(12, 123, 147, 0.12);
        --surface-border: rgba(127, 127, 127, 0.16);
        --surface-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
        --surface-shadow-hover: 0 18px 34px rgba(15, 23, 42, 0.12);
    }

    [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background:
            radial-gradient(circle at top right, rgba(12, 123, 147, 0.08), transparent 28%),
            radial-gradient(circle at top left, rgba(20, 93, 160, 0.08), transparent 22%);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--surface-border);
    }

    [data-testid="stSidebarNav"] a {
        font-weight: 500;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        color: var(--accent) !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] img {
        border-radius: 50%;
        border: 3px solid rgba(12, 123, 147, 0.45);
        padding: 3px;
    }

    .hero-shell,
    .surface-card,
    .project-card,
    .proof-card,
    .mini-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--surface-border);
        box-shadow: var(--surface-shadow);
        border-radius: 18px;
    }

    .hero-shell {
        padding: 1.7rem;
        margin-bottom: 1.1rem;
        position: relative;
        overflow: hidden;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -80px -80px auto;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(12, 123, 147, 0.12), transparent 62%);
        pointer-events: none;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.8rem;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-size: 2.35rem;
        line-height: 1.02;
        letter-spacing: -0.04em;
        font-weight: 800;
        margin-bottom: 0.9rem;
        max-width: 1000px;
    }

    .hero-subtitle {
        font-size: 1.8rem;
        line-height: 1.7;
        opacity: 0.86;
        max-width: 800px;
        margin-bottom: 0;
    }

    .proof-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.4rem;   
    margin-top: 0.3rem;
}

    .hero-proofline {
        margin-top: 1rem;
        font-size: 0.95rem;
        font-weight: 600;
        opacity: 0.88;
    }

    .proof-card,
    .mini-card {
        padding: 1.2rem 1.2rem;
        height: 100%;
    }

    .proof-label,
    .mini-label,
    .project-topline {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0.3rem;
    }

    .proof-value {
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }

    .proof-copy,
    .mini-copy {
        font-size: 0.92rem;
        opacity: 0.8;
        line-height: 1.55;
        margin-bottom: 0;
    }

    .section-head {
        margin: 2.4rem 0 1rem 0;
    }

    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .section-title {
        font-size: 1.9rem;
        line-height: 1.1;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.03em;
    }

    .section-copy {
        font-size: 1rem;
        opacity: 0.82;
        max-width: 840px;
        margin-bottom: 0;
    }

    .surface-card {
        padding: 1.2rem 1.25rem;
        height: 100%;
    }

    .project-card {
        padding: 1.25rem;
        height: 100%;
        transition: 0.25s ease;
        margin-bottom: 0.75rem;
    }

    .project-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--surface-shadow-hover);
    }

    .project-title {
        font-size: 1.2rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0.1rem 0 0.8rem 0;
    }

    .project-row {
        margin-bottom: 0.7rem;
    }

    .project-row p {
        margin: 0.18rem 0 0 0;
        line-height: 1.62;
        opacity: 0.85;
        font-size: 0.94rem;
    }

    .project-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        color: var(--accent);
    }

    .project-result {
        border: 1px solid var(--surface-border);
        background: var(--accent-soft);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin: 0.9rem 0 0.8rem 0;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .tag-row {
        margin-top: 0.6rem;
    }

    .tag {
        display: inline-block;
        padding: 0.28rem 0.74rem;
        margin: 0.18rem 0.2rem 0 0;
        border-radius: 999px;
        border: 1px solid rgba(12, 123, 147, 0.24);
        background: rgba(12, 123, 147, 0.08);
        color: var(--text-color);
        font-size: 0.75rem;
        font-weight: 700;
    }

    .capability-list {
        display: grid;
        gap: 0.8rem;
    }

    .capability-item {
        border: 1px solid var(--surface-border);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        background: rgba(127, 127, 127, 0.05);
    }

    .capability-item h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
    }

    .capability-item p {
        margin: 0;
        font-size: 0.93rem;
        opacity: 0.82;
        line-height: 1.55;
    }

    .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    .profile-name {
        text-align: center;
        margin: 1rem 0 0.2rem 0;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .profile-role {
        text-align: center;
        margin-top: 0;
        opacity: 0.82;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .footer-wrap {
        text-align: center;
        padding: 1.8rem 0 0.7rem 0;
    }

    .footer-links {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem 1.4rem;
        margin: 1rem 0 0.8rem 0;
    }

    .footer-link {
        color: var(--accent);
        text-decoration: none;
        font-weight: 700;
    }

    .gallery-caption {
        font-size: 0.9rem;
        opacity: 0.78;
        margin-top: 0.6rem;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 2.2rem; }
        .proof-grid { grid-template-columns: 1fr; }
        .section-title { font-size: 1.55rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.write("")
    if profile_pic_path.exists():
        _, center_col, _ = st.columns([0.5, 2, 0.5])
        with center_col:
            st.image(str(profile_pic_path), width=180)
    else:
        st.markdown(
            """
            <div style="width:160px;height:160px;border-radius:50%;
                        background:linear-gradient(135deg,#0c7b93,#145da0);
                        display:flex;align-items:center;justify-content:center;
                        color:#fff;font-size:2.3rem;margin:0 auto;">
                GT
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h1 class='profile-name'>Ganesh Todkari</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='profile-role'>MBA-IT Candidate | Data Analytics, Business Analysis, and Applied ML</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown(
        """
        <div style="font-size:0.92rem; line-height:1.75;">
            <div><a href="mailto:ganesh697todkari@gmail.com" style="color:inherit; text-decoration:none;">Email</a></div>
            <div><a href="https://linkedin.com/in/GaneshTodkari" style="color:inherit; text-decoration:none;">LinkedIn</a></div>
            <div><a href="https://github.com/GaneshTodkari" style="color:inherit; text-decoration:none;">GitHub</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if resume_path.exists():
        with open(resume_path, "rb") as pdf_file:
            st.download_button(
                label="Download Resume",
                data=pdf_file,
                file_name="Ganesh_Todkari_Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )


st.markdown(
    """
    <div class="hero-shell">
        <div class="eyebrow">Data + Business Portfolio</div>
        <div class="hero-title">At the intersection of Computer Science and Business Strategy, I design analytics and ML products that turn raw data into actionable outcomes.</div>
        <p class="hero-subtitle">
            I am a Computer Science graduate currently pursuing an MBA in IT. I work across Data Analytics,
            Business Analysis, Data Science, and Machine Learning to solve business problems through dashboards,
            forecasting systems, fraud analytics, NLP products, and process redesign.
        </p>
        <div class="hero-proofline">I build systems, not just models: live APIs, interactive Streamlit products, forecasting, and AI Resume Match system.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.2, 2.6], gap="medium")

with left_col:
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    if st.button("🚀 View Projects", key="hero_projects", type="primary", use_container_width=True):
        st.switch_page("pages/1_Retail_Demand_Forecasting.py")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if resume_path.exists():
        with open(resume_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Resume",
                data=pdf_file,
                file_name="Ganesh_Todkari_Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


with right_col:
    st.markdown(
        """
        <div class="proof-grid">
            <div class="proof-card">
                <div class="proof-label">Hybrid Delivery</div>
                <div class="proof-value">Analytics + Business Context</div>
                <p class="proof-copy">Projects connect technical execution with stakeholder needs and operational decisions.</p>
            </div>
            <div class="proof-card">
                <div class="proof-label">Interactive Systems</div>
                <div class="proof-value">Try the Models</div>
                <p class="proof-copy">I design interactive ML and analytics products with APIs, interfaces, and decision-ready outputs.</p>
            </div>
            <div class="proof-card">
                <div class="proof-label">Execution</div>
                <div class="proof-value">Insights to Action</div>
                <p class="proof-copy">I build dashboards, APIs, and decision tools that help teams move from analysis to action.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Value Proposition</div>
        <div class="section-title">A hybrid profile across analytics, business understanding, and technical execution</div>
        <p class="section-copy">
            Why this matters: I can analyze the problem, understand the business context, and build the system that supports the decision.
        </p>
        <p class="section-copy" style="margin-top:0.55rem;">
            I position data work in business terms. That means understanding the operating problem, structuring the data,
            selecting the right level of analysis or modeling, and presenting outputs in a form decision-makers can use.
            The focus is business outcomes first, with technical depth supporting that outcome.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

value_cols = st.columns(3, gap="medium")
value_cards = [
    {
        "label": "Data to Decisions",
        "copy": "I translate data into business outcomes through analytics, dashboards, predictive systems, and structured recommendations.",
    },
    {
        "label": "Business + Technical Understanding",
        "copy": "My MBA-IT training helps me bridge stakeholder needs, process thinking, and technical implementation in the same solution.",
    },
    {
        "label": "End-to-End Execution",
        "copy": "I work across analytics, business analysis, and ML systems, shaping outputs into interfaces and insights that are practical to use.",
    },
]
for col, card in zip(value_cols, value_cards):
    with col:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">{card['label']}</div>
                <p class="mini-copy">{card['copy']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )




featured_projects = [
    {
        "category": "Data Science | Forecasting",
        "title": "Retail Demand Forecasting",
        "problem": "Store managers need reliable demand estimates to plan inventory, staffing, and promotions.",
        "solution": "Built a Rossmann forecasting system using XGBoost, calendar feature engineering, FastAPI inference, and a manager-facing Streamlit dashboard.",
        "result": "Validation R2 of 0.85 and a product-style planning interface with scenario simulation and forecast interpretation.",
        "skills": ["Python", "XGBoost", "FastAPI", "Streamlit"],
        "link": "pages/1_Retail_Demand_Forecasting.py",
        "button": "Open Forecasting Product",
    },
    {
        "category": "Data Science | Risk Analytics",
        "title": "Credit Card Fraud Detection",
        "problem": "Fraud teams need to detect rare risky transactions without overwhelming investigations with false positives.",
        "solution": "Designed an anomaly detection workflow for imbalanced data using SMOTE, feature preparation, and a deployed scoring interface.",
        "result": "System designed to support precision-driven risk review.",
        "skills": ["Scikit-Learn", "SMOTE", "XGBoost", "Risk Analytics"],
        "link": "pages/2_Credit_Card_Security_Analysis.py",
        "button": "Open Fraud Case Study",
    },
    {
        "category": "NLP | Interactive Scoring",
        "title": "Resume Match System",
        "problem": "Recruiters and candidates need a faster way to evaluate role fit beyond manual keyword review.",
        "solution": "I built an NLP workflow that parses resumes and job descriptions, scores semantic and skill alignment, and shows matched versus missing capabilities.",
        "result": "A reusable interactive scoring system with semantic, skill, and experience signals surfaced through a portfolio product.",
        "skills": ["NLP", "Semantic Similarity", "Skill Extraction", "Streamlit"],
        "link": "pages/6_Resume_Match_System.py",
        "button": "Open NLP Product",
    },
    {
        "category": "Data Analytics | BI",
        "title": "Insurance Fraud Intelligence Dashboard",
        "problem": "Analysts need to reduce manual review time while prioritizing risky claims faster.",
        "solution": "Created a Power BI dashboard that highlights suspicious patterns, segments, and investigator priorities using structured fraud analysis.",
        "result": "Reduced investigation workflow time by 60 percent through clearer monitoring and prioritization.",
        "skills": ["Power BI", "SQL", "DAX", "Data Modeling"],
        "link": "pages/3_Insurance_Fraud_Claim_Analysis.py",
        "button": "Open Dashboard Case Study",
    },
    {
        "category": "Business Analysis | Process Improvement",
        "title": "VybeRiders Process Automation",
        "problem": "Manual checkout was slow, error-prone, and dependent on staff calculations.",
        "solution": "I redesigned the workflow into a digital admin system with automated billing, ride tracking, and payment handoff.",
        "result": "<strong>95% faster checkout:</strong> reduced from 10 minutes to 15 seconds.",
        "skills": ["Requirement Gathering", "Process Mapping", "BPMN", "Automation Design"],
        "link": "pages/7_VybeRiders_Admin_System.py",
        "button": "Open BA Case Study",
    },
]



st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Capabilities</div>
        <div class="section-title">Core capabilities across analytics, business, and ML</div>
    </div>
    """,
    unsafe_allow_html=True,
)

cap_col1, cap_col2 = st.columns(2, gap="large")
with cap_col1:
    st.markdown(
        """
        <div class="surface-card capability-list">
            <div class="capability-item">
                <h4>Data Analytics</h4>
                <p>I build dashboards and analytical outputs that help teams monitor performance, identify trends, and support better decisions.</p>
            </div>
            <div class="capability-item">
                <h4>Data Science and Machine Learning</h4>
                <p>I design predictive systems for forecasting and risk analysis, then expose them through interfaces people can actually use.</p>
            </div>
            <div class="capability-item">
                <h4>NLP and Intelligent Scoring</h4>
                <p>I create intelligent scoring workflows that combine parsing, semantic similarity, and explainable outputs for real-world tasks.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cap_col2:
    st.markdown(
        """
        <div class="surface-card capability-list">
            <div class="capability-item">
                <h4>Business Analysis and Process Design</h4>
                <p>I analyze workflows, gather requirements, and redesign business processes to reduce friction and improve operational efficiency.</p>
            </div>
            <div class="capability-item">
                <h4>System Thinking</h4>
                <p>I build systems, not just models, by connecting data, interfaces, APIs, and business context into one usable solution.</p>
            </div>
            <div class="capability-item">
                <h4>Decision-Focused Delivery</h4>
                <p>I turn analysis into clear recommendations, measurable outcomes, and interfaces that support real business decisions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="section-head">
        <div class="section-label">How I Work</div>
        <div class="section-title">A hybrid approach across business and technical execution</div>
        <p class="section-copy">
            Why this matters: it shows how I move from business question to usable output instead of stopping at analysis alone.
        </p>
        <p class="section-copy" style="margin-top:0.55rem;">
            I approach projects as business decision systems: understand the operating need, analyze the data, build the right technical solution,
            and deliver outputs in a form that stakeholders can trust and use.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

work_cols = st.columns(3, gap="medium")
work_items = [
    {
        "title": "1. Understand the business need",
        "copy": "I start from the problem, the process, and the stakeholder need before choosing the right analytics or modeling approach.",
    },
    {
        "title": "2. Build the right data solution",
        "copy": "Depending on the problem, that may mean dashboarding, exploratory analysis, predictive modeling, or process redesign.",
    },
    {
        "title": "3. Present outputs for action",
        "copy": "I package insights into dashboards, interfaces, and recommendations so the work supports action, not just analysis.",
    },
]
for col, item in zip(work_cols, work_items):
    with col:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">{item['title']}</div>
                <p class="mini-copy">{item['copy']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")
st.markdown(
    """
    <div class="footer-wrap">
        <h4 style="margin-bottom:0.3rem;">Actively seeking Data Analyst, Business Analyst, and Data Science opportunities</h4>
        <p style="margin-bottom:0.45rem; opacity:0.82;">
            If you are hiring for roles that value analysis, business understanding, and technical execution, let’s connect.
        </p>
        <div class="footer-links">
            <a href="mailto:ganesh697todkari@gmail.com" class="footer-link">Email</a>
            <a href="https://linkedin.com/in/GaneshTodkari" class="footer-link">LinkedIn</a>
            <a href="https://github.com/GaneshTodkari" class="footer-link">GitHub</a>
        </div>
        <p style="font-size:0.76rem; opacity:0.7;">Copyright 2026 Ganesh Todkari</p>
    </div>
    """,
    unsafe_allow_html=True,
)
