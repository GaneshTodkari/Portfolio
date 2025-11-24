import streamlit as st
import os

st.set_page_config(page_title="Contoso Retail Analysis", page_icon="🏬", layout="wide")

st.title("🏬 Contoso Retail Analysis")
st.markdown("""
**Project Overview:**
A comprehensive analysis of the Contoso retail dataset using Power BI. 
Key metrics include sales performance by region, product category trends, and customer demographics.
""")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Images are in the main 'assets' folder, not 'fraud_car'
assets_dir = os.path.join(current_dir, "..", "assets","contoso") 
pbix_path = os.path.join(assets_dir, "contoso.pbix")

# --- DASHBOARD GALLERY ---
st.markdown("### 🖼️ Dashboard Insights")

# Create tabs for organized viewing
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Sales Overview", "Product Analysis", "Customer Demographics", ])

# --- TAB 1: SALES OVERVIEW ---
with tab4:
    col1, col2 = st.columns([2, 1])
    with col1:
        img_path = os.path.join(assets_dir, "dashbord_1.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Sales Dashboard")
        else:
            st.error("Missing: product.png (Please save your screenshot in 'assets/')")
    
    with col2:
        img_path = os.path.join(assets_dir, "dashbord_2.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Sales Dashboard")
        else:
            st.error("Missing: product2.png (Please save your screenshot in 'assets/')")
with tab1:
    # Using your uploaded sales dashboard image
    col1, col2 = st.columns([2, 1])
    with col1:
        img_path = os.path.join(assets_dir, "product.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Sales Dashboard: Revenue & Category Performance", use_column_width=True)
        else:
            st.error("Missing: product.png (Please save your screenshot in 'assets/')")
    
    with col2:
        img_path = os.path.join(assets_dir, "product2.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Sales Dashboard: Revenue & Category Performance", use_column_width=True)
        else:
            st.error("Missing: product2.png (Please save your screenshot in 'assets/')")

# --- TAB 2: PRODUCT ANALYSIS ---
with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        img_path = os.path.join(assets_dir, "sub (1).png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Product Category Breakdown", use_column_width=True)
        else:
            st.error("Missing: product.png ")
    
    with col2:
        img_path = os.path.join(assets_dir, "sub (2).png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Product Category Breakdown", use_column_width=True)
        else:
            st.error("Missing: product2.png (Please save your screenshot in 'assets/')")

# --- TAB 3: CUSTOMER DEMOGRAPHICS ---
with tab3:
    col1, col2 = st.columns([2, 1])
    with col1:
        img_path = os.path.join(assets_dir, "geo (1).png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Demographic insights", use_column_width=True)
        else:
            st.error("Missing: product.png ")
    
    with col2:
        img_path = os.path.join(assets_dir, "geo (2).png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Ddemographic insights", use_column_width=True)
        else:
            st.error("Missing: product2.png (Please save your screenshot in 'assets/')")


st.divider()

# --- DOWNLOAD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.info("""
    ℹ️ **Note:** This analysis was built using **Power BI Desktop**. 
    You can download the source file to explore the data model and DAX measures.
    """)

with col2:
    if os.path.exists(pbix_path):
        with open(pbix_path, "rb") as f:
            st.download_button(
                label="📥 Download .pbix Source File",
                data=f,
                file_name="Contoso_Retail.pbix",
                mime="application/octet-stream"
            )
    else:
        st.error("⚠️ Missing file: assets/contoso.pbix")