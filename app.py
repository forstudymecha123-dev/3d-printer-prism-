"""
app.py  —  SENA: Smart Equipment Navigator & Assistant
Custom branding for ProtoForge AI 2026 Hackathon
Secure RAG + Gemini 3.1 Flash Lite implementation
"""

import streamlit as st
from rag_pipeline import PRISMRagPipeline # Keeping the file link for now

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SENA · Lab Sentinel",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for the SENA Dark Mode Glow-up ──────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --orange:  #ff6b2b; /* Hot Nozzle */
    --cyan:    #00d4e8; /* LCD Blue */
    --base:    #0b0c10;
    --card:    #1c1f2b;
    --text:    #eef0f8;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--base) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
}

/* Glassmorphism Chat Bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-bottom: 15px !important;
}

/* User Bubble Accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border: 1px solid #ff6b2b44 !important;
}

/* Assistant Bubble Accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 4px solid var(--cyan) !important;
}

.sena-header { padding: 1.5rem 0; text-align: center; }
.sena-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b2b, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── SECURE KEY RETRIEVAL ──────────────────────────────────────
def get_gemini_key():
    try:
        # Masked key retrieval from Streamlit Secrets
        return st.secrets["GEMINI_KEY"]
    except Exception:
        return ""

api_key = get_gemini_key()

# ── SENA Header ──────────────────────────────────────────────
st.markdown("""
<div class="sena-header">
    <div class="sena-title">🤖 SENA</div>
    <div style="color: #4a5272; font-family: 'JetBrains Mono'; letter-spacing: 2px;">
        SMART EQUIPMENT NAVIGATOR & ASSISTANT
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configuration ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦾 SENA Control Panel")
    if api_key:
        st.success("✅ Gemini Key Securely Loaded")
    else:
        st.error("🔑 KEY MISSING: Add 'GEMINI_KEY' to Streamlit Secrets.")
    
    st.divider()
    uploaded_pdf = st.file_uploader("📂 Upload Lab Manual (PDF)", type=["pdf"])

# ── Chat Engine ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "content": "Yo! SENA is online. Which machine are we calibrating today, bruv? 🏎️💨"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask SENA about 3D printing, CNC, or Laser safety..."):
    if not api_key:
        st.error("Bruv, I need the API key in the secrets to wake up.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize the RAG Pipeline
    pipeline = PRISMRagPipeline(api_key=api_key)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            # Running the RAG answer logic
            stream, used_rag, pages = pipeline.answer(
                query=prompt,
                history=st.session_state.messages[:-1],
                stream=True,
            )
            for chunk in stream:
                full_response += chunk.text
                placeholder.markdown(full_response + " ▌")
            placeholder.markdown(full_response)
            
            # Badge labeling
            if used_rag:
                st.info(f"📄 Source: Lab Manual (pp. {', '.join(map(str, pages))})")
            else:
                st.caption("🧠 Source: General Mechatronics Knowledge")
                
        except Exception as e:
            st.error(f"API caught an L: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
