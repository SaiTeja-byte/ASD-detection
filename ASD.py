import streamlit as st
if st.button("Go to Home"):
    st.switch_page("ASD.py")
# Title
st.title("Autism Spectrum Disorder Detection")

# Grid layout for category selection
col1, col2 = st.columns(2)

with col1:
    if st.button("Adolescents (13-17)"):
        st.switch_page("pages/Adolescents.py")

    if st.button("Toddlers (Up to 36 months)"):
        st.switch_page("pages/Toddlers.py")

with col2:
    if st.button("Adults (>18)"):
        st.switch_page("pages/Adults.py")

    if st.button("Children (4-12)"):
        st.switch_page("pages/Children.py")
