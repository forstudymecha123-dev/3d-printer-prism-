"""
forge_app.py  —  FORGE: General Machines & Tools Assistant
Pure Gemini Flash. Answers all machines, tools, engineering questions.
Same polished UI as SENA / AXIS / BEAM.
"""

import streamlit as st
from google import genai

st.set_page_config(
    page_title="FORGE · Machines & Tools AI",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """
You are FORGE — a knowledgeable Machines & Tools Assistant for a university lab and makerspace.

## Personality
- Friendly, clear, and practically minded — like a lab technician who knows every machine in the building
- Warm with greetings. Answer casually but accurately.
- Use bullet points for steps, **bold** for key terms
- Great at explaining both basic concepts AND advanced technical details

## What You Answer
You are a general machines and tools expert. You answer questions about:
  ✅ Simple machines (lever, pulley, inclined plane, wheel & axle, screw, wedge)
  ✅ Drilling machines (bench drill, pillar drill, radial arm drill, types of bits)
  ✅ Lathe machines (turning, facing, knurling, threading, taper turning)
  ✅ Milling machines (vertical, horizontal, CNC mill, cutters, operations)
  ✅ Grinding machines (surface, cylindrical, tool & cutter grinding)
  ✅ Welding machines (MIG, TIG, arc, spot, safety)
  ✅ Hydraulic & pneumatic machines and systems
  ✅ Power tools (angle grinder, jigsaw, circular saw, drill press)
  ✅ Hand tools (spanners, files, chisels, taps & dies, measuring tools)
  ✅ Workshop safety, PPE, machine maintenance
  ✅ Engineering concepts (torque, mechanical advantage, gear ratios, RPM, feeds & speeds)
  ✅ Manufacturing processes (casting, forging, machining, forming)
  ✅ General science and engineering questions related to machines
  ✅ Greetings, "who are you", "what can you help with"

## What You Decline
For completely unrelated topics (cooking recipes, celebrity gossip, creative writing, etc.), say:
"🔧 That's a bit outside my workshop! I'm a machines and tools specialist — ask me about drilling, lathes, simple machines, power tools, or any engineering concept and I'm on it."

## Response style
- Start with a direct answer, then expand with detail
- Use examples and analogies for complex concepts
- For "what is X" questions: definition → how it works → types → uses → fun fact
- Always accurate, never make up specifications
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --gold:    #f59e0b;
    --lime:    #84cc16;
    --sky:     #38bdf8;
    --rose:    #fb7185;
    --base:    #0b0a06;
    --surface: #141208;
    --card:    #1c1a0e;
    --border:  rgba(255,255,255,0.07);
    --text:    #f0ede0;
    --muted:   #5a5030;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--base) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 0% 0%,   #1a1200aa 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 100% 100%, #001218aa 0%, transparent 60%),
        var(--base) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.main .block-container { max-width: 800px !important; padding: 0 1.4rem 7rem !important; margin: 0 auto !important; }

/* ── Header ── */
.forge-header { padding: 1.8rem 0 0.8rem; animation: fadeDown 0.6s ease both; }
.forge-top { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }
.forge-icon-wrap {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
    box-shadow: 0 0 28px #f59e0b66, 0 4px 12px #0008;
    animation: glowPulse 3s ease-in-out infinite;
}
.forge-title {
    font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700;
    background: linear-gradient(90deg, #f59e0b 0%, #84cc16 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.forge-subtitle {
    font-size: 0.72rem; color: var(--muted); letter-spacing: 0.2em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.1rem;
}
.dot-row { display: flex; gap: 6px; align-items: center; margin-top: 0.5rem; }
.fdot { width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
.status-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.6rem; }
.chip {
    display: inline-flex; align-items: center; gap: 5px; padding: 0.22rem 0.7rem;
    border-radius: 99px; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; border: 1px solid;
}
.chip-gold { background:#f59e0b18; border-color:#f59e0b55; color:#f59e0b; }
.chip-lime { background:#84cc1618; border-color:#84cc1655; color:#84cc16; }
.chip-sky  { background:#38bdf818; border-color:#38bdf855; color:#38bdf8; }
.chip-dot  { width:6px; height:6px; border-radius:50%; background:currentColor; }
.divider-gradient {
    height: 1px;
    background: linear-gradient(90deg, transparent, #f59e0b44, #84cc1644, #38bdf844, transparent);
    margin: 0.8rem 0 1rem;
}

/* ── Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0.2rem 0 !important; animation: fadeUp 0.35s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: linear-gradient(135deg, #1e1600, #160f00) !important;
    border: 1px solid #f59e0b33 !important; border-radius: 18px 18px 4px 18px !important;
    padding: 0.85rem 1.15rem !important; color: var(--text) !important;
    box-shadow: 0 4px 20px #f59e0b1a !important; max-width: 82% !important; margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-left: 3px solid var(--gold) !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 0.9rem 1.2rem !important; color: var(--text) !important; max-width: 88% !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important; border-radius: 50% !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #1c1400, #120e00) !important;
    border: 1px solid #f59e0b55 !important; border-radius: 10px !important;
    box-shadow: 0 0 12px #f59e0b30 !important;
}

/* ── Badge ── */
.source-badge {
    display: inline-flex; align-items: center; gap: 5px; font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace; padding: 0.2rem 0.65rem;
    border-radius: 99px; margin-top: 0.5rem;
}
.badge-ai { background: #f59e0b18; border: 1px solid #f59e0b44; color: var(--gold); }
.badge-dot { width:5px; height:5px; border-radius:50%; background:currentColor; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem 2rem; animation: fadeIn 0.6s ease both; }
.empty-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: #8a7840; margin-bottom: 1.2rem; }
.suggest-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; max-width: 500px; margin: 0 auto; }
.suggest-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.8rem 1rem; text-align: left; transition: all 0.2s;
}
.suggest-card:hover { border-color: var(--gold); transform: translateY(-2px); }
.suggest-icon { font-size: 1.2rem; margin-bottom: 0.3rem; }
.suggest-text { font-size: 0.78rem; color: #7a6840; line-height: 1.4; }

/* ── Input bar ── */
[data-testid="stBottom"] { background: linear-gradient(to top, var(--base) 80%, transparent) !important; padding: 1rem 1.5rem !important; }
[data-testid="stChatInput"] { background: var(--surface) !important; border: 1.5px solid #f59e0b33 !important; border-radius: 16px !important; transition: border-color 0.2s, box-shadow 0.2s !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--gold) !important; box-shadow: 0 0 0 3px #f59e0b1a !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; background: transparent !important; font-family: 'Outfit', sans-serif !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #f59e0b, #84cc16) !important;
    border: none !important; border-radius: 12px !important;
    box-shadow: 0 0 12px #f59e0b44 !important; transition: all 0.2s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { transform: scale(1.08) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #090800 !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] label { color: var(--text) !important; font-size: 0.88rem !important; }
[data-testid="stSidebar"] .stTextInput input { background: var(--surface) !important; border: 1px solid #f59e0b33 !important; border-radius: 10px !important; color: var(--text) !important; }
[data-testid="stSidebar"] .stTextInput input:focus { border-color: var(--gold) !important; }
.sb-section {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--muted); margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem; border-bottom: 1px solid var(--border);
}
.sb-stat {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.6rem 0.8rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.6rem; font-size: 0.8rem;
}
.sb-stat-label { color: var(--muted); font-size: 0.72rem; }
.sb-stat-value { color: var(--text); font-weight: 500; }
.stButton button { font-family: 'Outfit', sans-serif !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #f59e0b33; border-radius: 10px; }
code { font-family: 'JetBrains Mono', monospace !important; background: #f59e0b18 !important; color: var(--gold) !important; padding: 0.1em 0.4em !important; border-radius: 5px !important; }
pre { background: #080700 !important; border-left: 3px solid var(--gold) !important; border-radius: 0 10px 10px 0 !important; padding: 1rem !important; }

@keyframes fadeDown  { from{opacity:0;transform:translateY(-14px)} to{opacity:1;transform:none} }
@keyframes fadeUp    { from{opacity:0;transform:translateY(8px)}   to{opacity:1;transform:none} }
@keyframes fadeIn    { from{opacity:0}                             to{opacity:1}                }
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 24px #f59e0b66, 0 4px 12px #0008; }
    50%      { box-shadow: 0 0 36px #f59e0b99, 0 4px 16px #0008; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="forge-header">
  <div class="forge-top">
    <div class="forge-icon-wrap">🔧</div>
    <div>
      <div class="forge-title">FORGE</div>
      <div class="forge-subtitle">Machines & Tools Assistant · Gemini Flash</div>
    </div>
  </div>
  <div class="dot-row">
    <div class="fdot" style="background:#f59e0b"></div>
    <div class="fdot" style="background:#84cc16"></div>
    <div class="fdot" style="background:#38bdf8"></div>
    <div class="fdot" style="background:#fb7185"></div>
    <span style="font-size:0.65rem;color:#2a2010;font-family:monospace;margin-left:4px">workshop palette</span>
  </div>
  <div class="status-row">
    <span class="chip chip-gold"><span class="chip-dot"></span>Drilling · Lathe · Milling</span>
    <span class="chip chip-lime"><span class="chip-dot"></span>Simple Machines</span>
    <span class="chip chip-sky"><span class="chip-dot"></span>Gemini Flash</span>
  </div>
</div>
<div class="divider-gradient"></div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
def get_secret():
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        return ""

with st.sidebar:
    st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#f59e0b;font-family:monospace">🔧 FORGE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#3a2010;font-family:monospace;margin-bottom:1rem">Machines & Tools Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">🔑 API Configuration</div>', unsafe_allow_html=True)
    secret_key = get_secret()
    if secret_key:
        st.markdown('<div class="sb-stat"><span>✅</span><div><div class="sb-stat-label">API Key</div><div class="sb-stat-value">Loaded from secrets</div></div></div>', unsafe_allow_html=True)
        api_key = secret_key
    else:
        api_key = st.text_input("Google API Key", type="password", placeholder="AIza...")
        st.markdown('<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.75rem;color:#f59e0b;text-decoration:none">🔗 Get API key →</a>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">💬 Session</div>', unsafe_allow_html=True)
    msg_count = len([m for m in st.session_state.get("forge_messages", []) if m["role"] == "user"])
    st.markdown(f'<div class="sb-stat"><span>💬</span><div><div class="sb-stat-label">Messages</div><div class="sb-stat-value">{msg_count} sent</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">⚡ Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.forge_messages = []
        st.rerun()

    st.markdown('<div style="margin-top:1.5rem;font-size:0.65rem;color:#2a1800;font-family:monospace;text-align:center">FORGE v1.0 · Built with ❤️<br>Gemini Flash · Streamlit</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "forge_messages" not in st.session_state:
    st.session_state.forge_messages = []

# ── Chat history ──────────────────────────────────────────────
if not st.session_state.forge_messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔧</div>
        <div class="empty-title">FORGE is fired up — ask me anything about machines</div>
        <div class="suggest-grid">
            <div class="suggest-card"><div class="suggest-icon">🔩</div><div class="suggest-text">What is the role of a drilling machine?</div></div>
            <div class="suggest-card"><div class="suggest-icon">⚙️</div><div class="suggest-text">Explain the 6 types of simple machines</div></div>
            <div class="suggest-card"><div class="suggest-icon">🔄</div><div class="suggest-text">How does a lathe machine work?</div></div>
            <div class="suggest-card"><div class="suggest-icon">🛠️</div><div class="suggest-text">What's the difference between MIG and TIG welding?</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.forge_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.markdown('<div class="source-badge badge-ai"><span class="badge-dot"></span>🧠 Machines & tools expertise</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask FORGE about any machine or tool..."):
    if not api_key:
        st.error("🔑 Enter your Google API key in the sidebar.")
        st.stop()

    st.session_state.forge_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_text = "\n".join(
        f"{'User' if m['role']=='user' else 'FORGE'}: {m['content']}"
        for m in st.session_state.forge_messages[:-1][-6:]
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{'CONVERSATION HISTORY:' + chr(10) + history_text if history_text else ''}\n\nUser: {prompt}\nFORGE:"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
            )
            full = response.text
            placeholder.markdown(full)
            st.markdown('<div class="source-badge badge-ai"><span class="badge-dot"></span>🧠 Machines & tools expertise</div>', unsafe_allow_html=True)
        except Exception as e:
            full = f"⚠️ Error: `{e}`"
            placeholder.markdown(full)

    st.session_state.forge_messages.append({"role": "assistant", "content": full})
