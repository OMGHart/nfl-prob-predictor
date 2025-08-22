# Streamlit Resume Page
import streamlit as st

st.title("Resume")
st.header("Professional Summary")
st.markdown("""
            Results-oriented data professional with an acute attention to detail and a passion for continuous learning.
            """)

st.divider()

st.header("Education")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
                **Eastern University** - *Master of Science in Data Science*
                """)
with col2:
    st.markdown("""
                (In Progress) 
                """)
    
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
                **Winthrop University** - *Bachelor of Science in Sociology*  
                <span style="margin-left:20px;"> Minor in statistics </span>
                """, unsafe_allow_html=True)
with col2:
    st.markdown("""
                (2009) 
                """)

st.divider()

st.header("Work Experience")

st.subheader("""
**Data Analyst** - *Journeyman Outfitter*,  2024-present 
""")
st.markdown("""
<ul style="margin-left:20px;">
    <li>Collect, clean, transform, and prepare data for analysis </li>
    <li>Develop statistical models to gain insights from data</li>
</ul>
            """, unsafe_allow_html=True)

st.subheader("""
**Reporting Lead** - *AT&T*,  2014-2024 
""")
st.markdown("""
<ul style="margin-left:20px;">
    <li>Developed and implemented dynamic calendar showing real-time
    forecast of scheduled installations at the team level</li> 
    <li>Analyze fallout and performance trends to recommend new processes
    and strategies</li> 
    <li>Partnered with cross-functional teams to ensure accurate and timely
    sales reporting </li> 
    <li>Generate funnels, drill-down reports, and visualizations to communicate
    key insights and recommendations </li> 
</ul>             
""", unsafe_allow_html = True)

st.divider()

st.header("""
Skills and Attributes 
             """)
st.markdown("""
<ul style="margin-left:20px;">
    <li>Data Science </li>
    <li>Data Analysis </li>
    <li>Statistics </li>
    <li>Python </li>
    <li>SQL </li>
    <li>R </li>
    <li>Time Magazine's 2006 Person of the Year</li>
</ul>
             """, unsafe_allow_html=True)

st.divider()

st.header("""
Certifications
             """)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    st.image("images/Google Data Analytics.png", caption="Google Data Analytics", width = 120) 

with col2:
    st.image("images/Google Advanced Data Analytics.png", caption="Google Advanced Data Analytics", width = 120) 
with col3:
    st.image("images/IBM Data Science.png", caption="IBM Data Science", width = 120)  
with col4:  
    st.image("images/SAS Programmer.png", caption="SAS Programmer", width = 120)  




