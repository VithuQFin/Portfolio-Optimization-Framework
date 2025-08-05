import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: left; font-size: 18px;'>
                Made with ❤️ by <br><strong>Vithusan Kailasapillai</strong><br>
                <a href="mailto:vithusan.kailasapillai@outlook.fr" style="text-decoration: none; color: #F63366;">Email</a><br>
                <a href="https://www.linkedin.com/in/vithusankailasapillai" target="_blank" style="text-decoration: none; color: #F63366;">LinkedIn</a><br>
                <a href="https://github.com/VithuQFin" target="_blank" style="text-decoration: none; color: #F63366;">GitHub</a>
            </div>
            """,
            unsafe_allow_html=True
        )
