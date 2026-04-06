"""
ui/auth.py
----------
Local-only login system using credentials.json.
MongoDB / SQL disabled.
"""

import hashlib
import json
import os
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def _verify(username, pw_hash):
    """
    Local-only authentication using credentials.json
    """
    creds_path = os.path.join(HERE, "credentials.json")
    if not os.path.exists(creds_path):
        return None

    creds = json.load(open(creds_path))
    user = creds["users"].get(username)

    if user and user["password"] == pw_hash:
        return user["name"], user["role"]

    return None


def login_page():
    _css = os.path.join(HERE, "style.css")
    if os.path.exists(_css):
        st.markdown(f"<style>{open(_css).read()}</style>", unsafe_allow_html=True)

    st.markdown('<div style="max-width:420px;margin:6rem auto 0">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:2rem">
            <p style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.15em;color:#475569;margin:0">
                Customer Intelligence Platform
            </p>
            <h1 style="font-size:1.75rem;font-weight:700;color:#f1f5f9;margin:0.5rem 0 0">
                Sign in
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card" style="padding:2rem">', unsafe_allow_html=True)
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pw")

    if st.button("Sign in", use_container_width=True):
        result = _verify(username.strip().lower(), _hash(password))
        if result:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username.strip().lower()
            st.session_state["name"] = result[0]
            st.session_state["role"] = result[1]
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#334155;font-size:0.75rem;margin-top:1.5rem">'
        'CLV Intelligence Platform — Team 6</p>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def logout():
    for k in ["authenticated", "username", "name", "role"]:
        st.session_state.pop(k, None)
    st.rerun()


def require_auth():
    if not st.session_state.get("authenticated"):
        login_page()
        st.stop()


def is_admin():
    return st.session_state.get("role") == "admin"

def current_role():
    return st.session_state.get("role", "viewer")
def current_user():
    return st.session_state.get("name", "Guest")