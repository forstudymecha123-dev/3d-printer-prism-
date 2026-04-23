"""
app.py — SENA 3D Printing Lab Assistant (Streamlit)
Uses pre-embedded ChromaDB for instant manual access
"""

import streamlit as st
from rag_pipeline_fixed import SENARagPipeline
import os


# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="SENA - 3D Lab Assistant",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0b0c10; }
    .stChatMessage { border-radius: 12px; }
    .response-box { 
        background: #13151c; 
        border-left: 4px solid #ff6b2b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-ready { color: #2ecc71; }
    .status-warning { color: #f39c12; }
    .status-error { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session ──────────────────────────────────────
@st.cache_resource
def init_pipeline():
    """Load pipeline once per session"""
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found in secrets.toml or env")
        st.stop()
    
    return SENARagPipeline(api_key=api_key, persist_dir="./sena_db")


pipeline = init_pipeline()


# ─── Sidebar: Status & Config ────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 API CONFIGURATION")
    
    status = pipeline.get_status()
    if status["ready"]:
        st.markdown(f"<p class='status-ready'>✅ Manual Loaded</p>", unsafe_allow_html=True)
        st.markdown(f"**Chunks:** {status['chunks_loaded']}")
    else:
        st.markdown(f"<p class='status-warning'>⚠️ No Manual Loaded</p>", unsafe_allow_html=True)
        st.info(
            "📚 Lab manual not found in ChromaDB.\n\n"
            "**Setup required:**\n"
            "```bash\npython setup_manual.py --pdf your_manual.pdf\n```\n"
            "Then restart the app."
        )
    
    st.markdown("### 📚 LAB MANUAL")
    if status["ready"]:
        st.markdown(f"<p>Database: <code>{status['db_path']}</code></p>", unsafe_allow_html=True)
        if st.button("🔄 Reload Pipeline"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.markdown("<p style='color:#f39c12;'>Waiting for setup...</p>", unsafe_allow_html=True)
    
    st.markdown("### 📊 DATABASE STATUS")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", status["chunks_loaded"])
    with col2:
        st.metric("Status", "Ready" if status["ready"] else "Empty")
    
    st.markdown("---")
    st.markdown("""
    **SENA** — 3D Printing Lab Assistant
    
    RAG + Gemini Flash • ChromaDB vector store
    
    ✅ FDM • SLA • SLS
    """)


# ─── Main Chat Interface ─────────────────────────────────────
st.markdown("""
<div style='text-align: center; margin: 2rem 0;'>
    <h1 style='color: #ff6b2b;'>🖨️ SENA</h1>
    <p style='color: #aaa; font-size: 14px;'>3D Printing Lab Assistant · RAG + Gemini Flash</p>
    <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;'>
        <span style='background: #13151c; padding: 6px 12px; border-radius: 20px; font-size: 12px;'>FDM · SLA · SLS</span>
        <span style='background: #13151c; padding: 6px 12px; border-radius: 20px; font-size: 12px;'>📚 RAG-powered</span>
        <span style='background: #13151c; padding: 6px 12px; border-radius: 20px; font-size: 12px;'>💾 ChromaDB</span>
        <span style='background: #13151c; padding: 6px 12px; border-radius: 20px; font-size: 12px;'>⚡ Gemini 1.5 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Chat History ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hey there! 👋 I'm SENA, your 3D printing lab assistant. What are you working on today, and how can I help you get that print dialed in? 🖨️"
        }
    ]


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── User Input & Response ──────────────────────────────────
user_query = st.chat_input("Ask SENA anything about 3D printing...")

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Get answer from RAG pipeline
            answer, used_rag, pages = pipeline.answer(user_query, stream=False)
            
            # Display response
            st.markdown(answer)
            
            # Show RAG metadata
            if used_rag:
                with st.expander("📄 Sources", expanded=False):
                    st.markdown(f"**Pages referenced:** {', '.join(map(str, pages)) if pages else 'N/A'}")
                    st.markdown("*Data sourced from lab manual via ChromaDB*")
            else:
                with st.expander("💡 Note", expanded=False):
                    st.markdown("*Response based on general 3D printing expertise (no manual data)*")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Try reloading the pipeline or check your API key.")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 12px; color: #888;'>
    SENA © University Makerspace · <a href='#' style='color: #ff6b2b;'>Report Issue</a>
</div>
""", unsafe_allow_html=True)


# Action buttons
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = [st.session_state.messages[0]]  # Keep intro
    st.rerun()
