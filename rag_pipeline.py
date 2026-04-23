import io, re
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import PyPDF2

SYSTEM_PROMPT = """
You are SENA — a specialist 3D Printing Lab Assistant.
Only answer based on the manual context provided. 
"""

class SENARagPipeline:
    def __init__(self, api_key: str, persist_dir: str = "./sena_db"):
        self.client = genai.Client(api_key=api_key)
        # Using the standard embedding function for Chroma
        self.emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/text-embedding-004" # Fixed model name
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="lab_manual",
            embedding_function=self.emb_fn
        )

    def ingest_pdf(self, pdf_bytes: bytes, source: str = "manual"):
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        chunks, metadatas, ids = [], [], []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip(): continue
            
            clean_text = re.sub(r'\s+', ' ', text).strip()
            chunks.append(clean_text)
            metadatas.append({"page": i + 1, "source": source})
            ids.append(f"{source}_p{i+1}")

        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def retrieve(self, query: str, k: int = 3):
        results = self.collection.query(query_texts=[query], n_results=k)
        return [
            {"content": doc, "page": meta["page"]} 
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def answer(self, query: str):
        chunks = self.retrieve(query)
        if not chunks:
            return "No relevant info found in the manual.", []
            
        context = "\n\n".join([f"[Page {c['page']}]: {c['content']}" for c in chunks])
        prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
        
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text, chunks
