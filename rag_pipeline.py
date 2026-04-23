"""
rag_pipeline.py  —  SENA: Smart Equipment Navigator & Assistant
RAG: PDF → chunks → ChromaDB → Gemini 3.1 Flash Lite
"""

import io, re
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import PyPDF2

SYSTEM_PROMPT = """
You are SENA — the Smart Equipment Navigator & Assistant for the REVA Mechatronics Lab.

## Personality
-  Mechatronics Expert who is polite.
- Friendly and hyped with greetings. If asked 'hi', welcome them to the lab!
- Use bullet points for steps, **bold** for key terms.

## Domain Rules
You ONLY answer questions about 3D printing, CNC, Laser cutting, and Lab safety.
For off-topic stuff, say: "🔧 That's outside my build plate, bruv! Stick to the hardware."
"""

class PRISMRagPipeline:
    def __init__(self, api_key: str, persist_dir: str = "./prism_db"):
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key, model_name="models/text-embedding-004",
        )
        self.collection = self.client.get_or_create_collection(
            name="sena_lab_manual",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        # THE FIX: USING THE 3.1 FLASH LITE MODEL
        self.model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=SYSTEM_PROMPT,
        )

    def extract_pdf(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]

    def chunk(self, text: str, size: int = 700, overlap: int = 120) -> list[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunks, cur, cur_len = [], [], 0
        for s in sents:
            if cur_len + len(s) > size and cur:
                chunks.append(" ".join(cur))
                ol, ol_len = [], 0
                for x in reversed(cur):
                    if ol_len + len(x) < overlap: ol.insert(0, x); ol_len += len(x)
                    else: break
                cur, cur_len = ol, ol_len
            cur.append(s); cur_len += len(s)
        if cur: chunks.append(" ".join(cur))
        return [c for c in chunks if len(c) > 60]

    def ingest_pdf(self, pdf_bytes: bytes, source: str = "lab_manual") -> int:
        pages = self.extract_pdf(pdf_bytes)
        ids, docs, metas = [], [], []
        for pg, text in pages:
            for j, c in enumerate(self.chunk(text)):
                ids.append(f"{source}_p{pg}_c{j}")
                docs.append(c)
                metas.append({"source": source, "page": pg})
        if not docs: return 0
        try:
            old = self.collection.get(where={"source": source})
            if old["ids"]: self.collection.delete(ids=old["ids"])
        except Exception: pass
        for i in range(0, len(docs), 100):
            self.collection.add(documents=docs[i:i+100], ids=ids[i:i+100], metadatas=metas[i:i+100])
        return len(docs)

    def has_manual(self) -> bool: return self.collection.count() > 0

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if not self.has_manual(): return []
        r = self.collection.query(query_texts=[query], n_results=min(k, self.collection.count()))
        return [{"content": doc, "page": r["metadatas"][0][i].get("page","?"), "dist": r["distances"][0][i]}
                for i, doc in enumerate(r["documents"][0]) if r["distances"][0][i] < 0.55]

    def answer(self, query: str, history: list[dict], stream: bool = True):
        chunks = self.retrieve(query)
        used_rag = bool(chunks)
        pages = sorted(set(c["page"] for c in chunks)) if chunks else []
        msg = f"LAB MANUAL EXCERPTS:\n{re.sub(r'[^\\x00-\\x7F]+', ' ', chunks[0]['content']) if used_rag else ''}\n\nQUESTION: {query}"
        hist = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in history]
        chat = self.model.start_chat(history=hist)
        if stream: return chat.send_message(msg, stream=True), used_rag, pages
        r = chat.send_message(msg)
        return r.text, used_rag, pages
