from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="VybeRiders Admin System",
    page_icon="🛵",
    layout="wide",
)

current_dir = Path(__file__).resolve().parent
assets_dir = current_dir.parent / "assets" / "vyberiders"
gallery_images = [path for path in sorted(assets_dir.glob("*.jpg")) if path.is_file()]
gallery_captions = [
    "Admin Login"
    "Admin dashboard for starting and managing trips",
    "Ride lifecycle handling for starting and managing trips",
    "Operational view supporting tracking and control",
    "Ride lifecycle handling for starting",
    "Ending ride, Automated billing and checkout workflow",
    "Records",
]

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 1rem;
    }

    .page-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.8rem;
        font-weight: 800;
        color: #0c7b93;
        margin-bottom: 0.55rem;
    }

    .hero-card,
    .case-card,
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(127, 127, 127, 0.16);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        border-radius: 18px;
    }

    .hero-card {
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.45rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0 0 0.8rem 0;
    }

    .hero-copy {
        font-size: 1.03rem;
        line-height: 1.7;
        opacity: 0.86;
        max-width: 820px;
        margin-bottom: 0;
    }

    .metric-card {
        padding: 1rem 1.05rem;
        height: 100%;
    }

    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 800;
        color: #0c7b93;
        margin-bottom: 0.28rem;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-copy {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.28rem;
        line-height: 1.5;
    }

    .section-head {
        margin: 2.2rem 0 0.9rem 0;
    }

    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.76rem;
        font-weight: 800;
        color: #0c7b93;
        margin-bottom: 0.3rem;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }

    .section-copy {
        font-size: 0.98rem;
        opacity: 0.82;
        max-width: 800px;
        margin-bottom: 0;
    }

    .case-card {
        padding: 1.15rem 1.2rem;
        height: 100%;
    }

    .case-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.02rem;
    }

    .case-card p,
    .case-card li {
        font-size: 0.95rem;
        line-height: 1.6;
        opacity: 0.86;
    }

    .tag-row {
        margin-top: 0.65rem;
    }

    .tag {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        margin: 0.18rem 0.22rem 0 0;
        border-radius: 999px;
        background: rgba(12, 123, 147, 0.08);
        border: 1px solid rgba(12, 123, 147, 0.22);
        color: var(--text-color);
        font-size: 0.75rem;
        font-weight: 700;
    }

    .flow-box {
        border: 1px solid rgba(127, 127, 127, 0.16);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        background: rgba(127, 127, 127, 0.05);
        height: 100%;
    }

    .flow-box h4 {
        margin-top: 0;
    }

    .flow-line {
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-kicker'>Real-World Project | Business Analysis | Product | System Design</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">VybeRiders Admin System</div>
        <p class="hero-copy">
            I designed this project as a business process re-engineering case study for a bike rental operation.
            The goal was to replace slow, manual checkout with a mobile-first admin workflow that automates ride management,
            billing, payment communication, and record storage.
        </p>
        <div class="tag-row">
            <span class="tag">Business Analysis</span>
            <span class="tag">Process Optimization</span>
            <span class="tag">Workflow Automation</span>
            <span class="tag">System Design</span>
            <span class="tag">Flutter</span>
            <span class="tag">Firebase</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3, gap="medium")
metrics = [
    ("Business Goal", "Digitize Checkout", "Replace a manual rental billing flow with a faster, more reliable operating process."),
    ("Measured Impact", "10 min to 15 sec", "Reduced checkout time by roughly 95 percent while improving billing consistency."),
    ("Role Coverage", "BA + Product + System", "I worked across process analysis, workflow design, business logic, and implementation."),
]
for col, (label, value, copy) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Case Study Overview</div>
        <div class="section-title">Business framing before technology</div>
        <p class="section-copy">
            This project is best understood as a business operations case study. The technology mattered,
            but the real value came from redesigning the workflow to reduce time, remove manual errors, and improve customer checkout.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns(2, gap="large")
with top_left:
    st.markdown(
        """
        <div class="case-card">
            <h4>Business Goal</h4>
            <p>
                Build a mobile-first admin system for a bike rental business that streamlines ride operations,
                automates billing, and improves the customer payment experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="case-card">
            <h4>Problem Statement</h4>
            <p>
                The existing process relied on manual time tracking, manual bill calculation, and separate payment handling.
                This created delays, increased the risk of billing errors, and made the checkout experience inefficient for both staff and customers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        """
        <div class="case-card">
            <h4>My Role</h4>
            <p>
                I worked across Business Analysis, Product Thinking, System Design, and Development. My primary focus was on understanding
                the operational problem, redesigning the workflow, defining the business logic, and translating that into a usable digital system.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="case-card">
            <h4>Why This Project Matters</h4>
            <p>
                It demonstrates process optimization, workflow automation, and system thinking in a real-world setting, making it highly relevant for
                Business Analyst, Product, Data Analyst, and operations-focused roles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">AS-IS vs TO-BE</div>
        <div class="section-title">Process redesign at the center of the solution</div>
        <p class="section-copy">
            The key shift was not just building software. It was moving from a manually dependent checkout flow to a structured, automated business process.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

flow_col1, flow_col2 = st.columns(2, gap="large")
with flow_col1:
    st.markdown(
        """
        <div class="flow-box">
            <h4>AS-IS Process</h4>
            <div class="flow-line">Manual billing -> time calculation -> cost calculation -> payment</div>
            <ul>
                <li>Ride duration had to be checked manually</li>
                <li>Billing depended on manual calculation</li>
                <li>Payment communication was disconnected from the checkout step</li>
                <li>Manual dependency increased operational friction and error risk</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with flow_col2:
    st.markdown(
        """
        <div class="flow-box">
            <h4>TO-BE Process</h4>
            <div class="flow-line">Start ride -> system tracks time -> auto billing -> WhatsApp payment -> database storage</div>
            <ul>
                <li>Ride lifecycle is captured digitally</li>
                <li>Billing is generated automatically from duration rules</li>
                <li>Receipts and UPI payment links are sent instantly</li>
                <li>Operational records are stored centrally in Firestore</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Solution</div>
        <div class="section-title">Business process automation through a mobile-first admin system</div>
        <p class="section-copy">
            I designed and developed the system around the business workflow, not just the screens. The application supports the full operational cycle from ride start to checkout completion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

solution_cols = st.columns(2, gap="large")
with solution_cols[0]:
    st.markdown(
        """
        <div class="case-card">
            <h4>Solution Overview</h4>
            <ul>
                <li><strong>Ride lifecycle management</strong>: start, end, and track rides in real time</li>
                <li><strong>Automated billing engine</strong>: calculates cost based on ride duration with defined rounding logic</li>
                <li><strong>Role-Based Access Control</strong>: secures admin-only actions using Firebase Authentication</li>
                <li><strong>Digital communication</strong>: sends receipts and payment links via WhatsApp</li>
                <li><strong>Centralized data layer</strong>: stores ride and transaction records in Firebase Firestore</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with solution_cols[1]:
    st.markdown(
        """
        <div class="case-card">
            <h4>Business Language Summary</h4>
            <ul>
                <li>Business process re-engineering for ride checkout</li>
                <li>Workflow automation to reduce manual effort</li>
                <li>Decision support through reliable operational records</li>
                <li>Process optimization to improve speed, consistency, and customer experience</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Impact</div>
        <div class="section-title">Operational efficiency with clear business value</div>
    </div>
    """,
    unsafe_allow_html=True,
)

impact_cols = st.columns(2, gap="large")
with impact_cols[0]:
    st.markdown(
        """
        <div class="case-card">
            <h4>Measured Result</h4>
            <p><strong>Reduced checkout time from 10 minutes to 15 seconds</strong>, which is approximately a 95 percent improvement in process efficiency.</p>
            <ul>
                <li>Eliminated manual billing errors</li>
                <li>Improved checkout consistency</li>
                <li>Reduced staff effort during ride completion</li>
                <li>Improved customer experience with faster payment support</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with impact_cols[1]:
    st.markdown(
        """
        <div class="case-card">
            <h4>Why Recruiters Should Care</h4>
            <p>
                This is not just a development project. It shows that I can identify an inefficient business process,
                redesign it, translate it into system logic, and deliver measurable real-world impact.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">BA Thinking</div>
        <div class="section-title">Concepts applied in the project</div>
    </div>
    """,
    unsafe_allow_html=True,
)

concepts_col, stories_col = st.columns(2, gap="large")
with concepts_col:
    st.markdown(
        """
        <div class="case-card">
            <h4>Key BA Concepts Applied</h4>
            <ul>
                <li>Business Process Re-engineering (BPR)</li>
                <li>Requirement Analysis</li>
                <li>Workflow Optimization</li>
                <li>Role-Based Access Control (RBAC)</li>
                <li>System Design Thinking</li>
                <li>Operational Efficiency Improvement</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stories_col:
    st.markdown(
        """
        <div class="case-card">
            <h4>User Stories</h4>
            <ul>
                <li>As an admin, I want to start a ride so that customer usage can be tracked accurately.</li>
                <li>As an admin, I want automatic billing so that I avoid calculation errors and checkout delays.</li>
                <li>As a customer, I want quick checkout so that I can complete payment without waiting.</li>
                <li>As a business operator, I want ride and billing records stored centrally so that operations are traceable and manageable.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">Tech Stack</div>
        <div class="section-title">Implementation layer supporting the business workflow</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="case-card">
        <div class="tag-row">
            <span class="tag">Flutter</span>
            <span class="tag">Firebase Authentication</span>
            <span class="tag">Firebase Firestore</span>
            <span class="tag">WhatsApp Integration</span>
            <span class="tag">UPI Payment Support</span>
            <span class="tag">Mobile-First Admin UX</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-head">
        <div class="section-label">AI-Assisted Development</div>
        <div class="section-title">Professional positioning of AI usage</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="case-card">
        <p>
            This project included AI-assisted development using tools such as ChatGPT and Gemini for rapid prototyping and selected implementation support.
            My primary contribution was defining the business logic, redesigning the workflow, structuring the system behavior, and translating the operational
            problem into a usable product solution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if gallery_images:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-label">Screens and Flow</div>
            <div class="section-title">Project visuals in sequence</div>
            <p class="section-copy">
                These images are shown in the same sequence as stored in the project assets and help illustrate the workflow and interface thinking behind the system.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    first_row = st.columns(3, gap="medium")
    second_row = st.columns(3, gap="medium")
    all_cols = first_row + second_row
    for idx, image_path in enumerate(gallery_images[:6]):
        with all_cols[idx]:
            st.image(str(image_path), use_container_width=True)
            caption = gallery_captions[idx] if idx < len(gallery_captions) else f"Screen {idx + 1}"
            st.caption(f"Screen {idx + 1}: {caption}")

st.markdown("---")
st.markdown(
    """
    <div class="case-card">
        <h4>Summary</h4>
        <p>
            VybeRiders Admin System demonstrates business understanding, workflow redesign, system design thinking, and real-world impact.
            It is a strong example of how I approach projects as business solutions first, with technology serving the process improvement goal.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
