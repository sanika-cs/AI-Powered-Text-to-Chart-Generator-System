import streamlit as st

from database import init_db


st.set_page_config(
    page_title="Text to Chart App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# initialize DB
init_db()

# Set page configuration

# Main Title
st.title("Text to Chart App")
#st.markdown("Welcome to the **Text-to-Chart App**! ")

# Description
st.markdown("""

This app helps you:
-  **Clean your dataset**
-  **Generate charts from text**
-  **Visualize insights instantly**

Use the sidebar or buttons below to get started.

""")

# Navigation Button
st.markdown("---")
st.subheader("Get Started")
if st.button("Go to Data Uploading and Cleaning"):
    st.switch_page("pages/Data_uploading_and_Cleaning.py")



st.subheader("Supported Charts ")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#####  1. Bar & Categorical Charts")
    st.markdown("""
- Bar Chart  
- Grouped Bar Chart  
- Stacked Bar Chart  
- Horizontal Bar Chart
""")

    st.markdown("#####  2. Line & Area Charts")
    st.markdown("""
- Line Chart  
- Multi-Line Chart  
- Step Line Chart  
- Area Chart  
- Stacked Area Chart
""")

with col2:
    st.markdown("#####  3. Distribution & Summary Charts")
    st.markdown("""
- Histogram  
- Box Plot  
- Violin Plot  
- Pie Chart  
- Donut Chart
""")

    st.markdown("#####  4. Business & KPI Charts")
    st.markdown("""
- Treemap  
- Waterfall Chart  
- Funnel Chart  
- Gauge Chart  
- KPI Card / Metric  
- Combo Chart (Bar + Line)
""")

with col3:
    st.markdown("#####  5. Correlation & Relationships")
    st.markdown("""
- Scatter Plot  
- Bubble Chart  
- Correlation Heatmap
""")

    st.markdown("#####  6. Geographic & Map Visuals")
    st.markdown("""
-  Choropleth Map  """)
