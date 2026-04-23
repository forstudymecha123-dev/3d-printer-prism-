# 🖨️ PRISM — 3D Printing Lab Assistant

> RAG-powered chatbot | Gemini 1.5 Flash | ChromaDB | Streamlit

PRISM is a domain-specific AI assistant for 3D printing labs. It answers questions using your uploaded lab manual PDF (via ChromaDB vector search) and falls back to Gemini Flash's general knowledge. It politely declines off-topic questions.

---

## 🗂️ Project Structure

```
prism/
├── app.py                        ← Streamlit UI
├── rag_pipeline.py               ← ChromaDB + Gemini RAG engine
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← Excludes secrets + DB
└── .streamlit/
    ├── config.toml               ← Theme config (dark + orange)
    └── secrets.toml.example      ← Template (don't commit actual secrets)
```

---

## ⚡ Local Setup

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/prism.git
cd prism

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (local only, never commit this)
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "AIza_YOUR_KEY_HERE"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

---

## 🚀 Deploy on Streamlit Cloud (via GitHub)

### Step 1 — Push to GitHub

```bash
# Inside your prism/ folder
git init
git add .
git commit -m "🖨️ initial commit: PRISM 3D printing assistant"

# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/prism.git
git branch -M main
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/prism`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Advanced settings"** → paste your secret:
   ```
   GOOGLE_API_KEY = "AIza_YOUR_ACTUAL_KEY"
   ```
6. Click **Deploy** 🚀

### Step 3 — Upload your PDF

Once deployed, open the app → sidebar → upload your lab manual PDF. ChromaDB will index it on first upload.

> ⚠️ Note: ChromaDB persists in `./prism_db/`. On Streamlit Cloud, this resets on each reboot. Users will need to re-upload the PDF after a cold start (or use a cloud DB like Pinecone for permanent storage).

---

## 🔄 Update the App (push changes)

```bash
git add .
git commit -m "✨ update: improve UI colors"
git push
```
Streamlit Cloud auto-redeploys on every push to `main`. 🎉

---

## 🧠 How RAG Works

```
PDF Upload
  → Extract text (PyPDF2)
  → Chunk (700 chars, 120 overlap)
  → Embed (Google text-embedding-004)
  → Store in ChromaDB

User Question
  → Embed query
  → Cosine similarity search (threshold < 0.55)
  → Inject top-k chunks into Gemini prompt
  → Stream answer with source badge (📄 manual / 🧠 general)
```

---

## 🎨 Color Palette

| Role | Color | Meaning |
|------|-------|---------|
| Primary / User | `#ff6b2b` | Hot nozzle orange |
| Assistant / RAG | `#00d4e8` | LCD display cyan |
| Success / Online | `#a3e635` | Extrusion lime |
| Highlight | `#f472b6` | Filament pink |
| General AI | `#a78bfa` | Knowledge violet |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `google-generativeai` | Gemini Flash + embeddings |
| `chromadb` | Vector database |
| `PyPDF2` | PDF text extraction |
