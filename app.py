"""
app.py  —  SENA: Smart Equipment Navigator & Assistant
Polished Streamlit UI + RAG + Gemini 3.1 Flash Lite
"""

import streamlit as st
from rag_pipeline import PRISMRagPipeline

st.set_page_config(page_title="SENA · Lab Assistant", page_icon="🤖", layout="centered")

# Use your previous CSS block here (omitted for brevity, keep the CSS you have)
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # PASTE YOUR CSS HERE

def get_api_key_from_secrets():
    try: return st.secrets.get("GEMINI_KEY", "") # SWAPPED TO YOUR MASKED NAME
    except Exception: return ""

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="prism-header">
  <div class="prism-top">
    <div class="prism-icon-wrap">🤖</div>
    <div class="prism-title-block">
      <div class="prism-title">SENA</div>
      <div class="prism-subtitle">Smart Equipment Navigator · Phase 2 Live</div>
    </div>
  </div>
</div>
<div class="divider-gradient"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#ff6b2b;font-family:monospace">⬡ SENA</div>', unsafe_allow_html=True)
    api_key = get_api_key_from_secrets()
    if api_key: st.success("✅ Gemini Key Securely Loaded")
    else: api_key = st.text_input("Google API Key", type="password")
    
    uploaded_pdf = st.file_uploader("Upload Lab Manual", type=["pdf"])
    if uploaded_pdf and api_key:
        pipeline = PRISMRagPipeline(api_key=api_key)
        n = pipeline.ingest_pdf(uploaded_pdf.read(), source=uploaded_pdf.name)
        st.session_state["pdf_loaded"] = True
        st.session_state["pdf_chunks"] = n

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Chat Loop
if not st.session_state.messages:
    st.markdown('<div class="empty-state"><div class="empty-title">SENA is online. What are we building, bruv?</div></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask SENA anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    pipeline = PRISMRagPipeline(api_key=api_key)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        stream, used_rag, pages = pipeline.answer(query=prompt, history=st.session_state.messages[:-1], stream=True)
        for chunk in stream:
            full += chunk.text
            placeholder.markdown(full + " ▌")
        placeholder.markdown(full)
        if used_rag: st.info(f"📄 Lab manual (pp. {', '.join(map(str, pages))})")

    st.session_state.messages.append({"role": "assistant", "content": full, "rag_used": used_rag, "pages": pages})
