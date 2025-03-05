# from flask import Flask, request, jsonify
# import os
# import re
# from PyPDF2 import PdfReader
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# import faiss
# import numpy as np

# app = Flask(__name__)

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# index = None  # Global FAISS index

# @app.route("/upload", methods=["POST"])
# def upload_pdf():
#     global index
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
    
#     # Save file temporarily
#     temp_dir = os.path.join(os.getcwd(), "temp")  # Create a temp folder in the current directory
#     os.makedirs(temp_dir, exist_ok=True)  # Ensure the folder exists
#     filepath = os.path.join(temp_dir, file.filename)

#     file.save(filepath)
    
#     # Extract text from PDF
#     pdfreader = PdfReader(filepath)
#     raw_text = "".join(page.extract_text() or "" for page in pdfreader.pages)
    
#     # Clean text
#     clean_text = re.sub(r'[^\w\s]', '', raw_text)
    
#     # Split text
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
#     texts = text_splitter.split_text(clean_text)
    
#     documents = [Document(page_content=text) for text in texts]

#     # Initialize FAISS with required arguments
#     embeddings = embedding_model.embed_documents(texts)
#     dimension = len(embeddings[0])  # Get the embedding dimension
#     faiss_index = faiss.IndexFlatL2(dimension)
#     faiss_index.add(np.array(embeddings, dtype=np.float32))

#     index = FAISS(
#         embedding_function=embedding_model,
#         index=faiss_index,
#         docstore=InMemoryDocstore(dict(enumerate(documents))),
#         index_to_docstore_id={i: i for i in range(len(documents))}
#     )

#     return jsonify({"message": "PDF processed and indexed successfully", "chunks": len(texts)})

# @app.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         data = request.get_json()
        
#         # Handle missing JSON
#         if data is None:
#             return jsonify({"error": "Invalid JSON. Ensure 'Content-Type' is 'application/json'."}), 400

#         query_text = data.get("query", "")
        
#         if not query_text:
#             return jsonify({"error": "Query text is missing in the request."}), 400

#         return jsonify({"response": f"Processing query: {query_text}"})
    
#     except Exception as e:
#         return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# # @app.route("/query", methods=["POST"])
# # def query_pdf():
# #     global index
# #     if not index:
# #         return jsonify({"error": "No PDF indexed yet"}), 400
    
# #     data = request.json
# #     query_text = data.get("query", "")
    
# #     if not query_text:
# #         return jsonify({"error": "No query provided"}), 400
    
# #     query_embedding = embedding_model.embed_query(query_text)
# #     distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k=5)
    
# #     return jsonify({"results": indices.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True)

#ver2
# from flask import Flask, request, jsonify
# import os
# import re
# from PyPDF2 import PdfReader
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# import faiss
# import numpy as np

# app = Flask(__name__)

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# index = None  # Global FAISS index
# documents = []  # To store processed document chunks

# @app.route("/upload", methods=["POST"])
# def upload_pdf():
#     global index, documents
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
    
#     # Save file temporarily
#     temp_dir = os.path.join(os.getcwd(), "temp")
#     os.makedirs(temp_dir, exist_ok=True)
#     filepath = os.path.join(temp_dir, file.filename)
#     file.save(filepath)
    
#     # Extract text from PDF
#     pdfreader = PdfReader(filepath)
#     raw_text = "".join(page.extract_text() or "" for page in pdfreader.pages)
    
#     # Clean text
#     clean_text = re.sub(r'[^\w\s]', '', raw_text)
    
#     # Split text into chunks
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
#     texts = text_splitter.split_text(clean_text)
    
#     documents = [Document(page_content=text) for text in texts]

#     # Create FAISS index
#     embeddings = embedding_model.embed_documents(texts)
#     dimension = len(embeddings[0])
#     faiss_index = faiss.IndexFlatL2(dimension)
#     faiss_index.add(np.array(embeddings, dtype=np.float32))

#     index = FAISS(
#         embedding_function=embedding_model,
#         index=faiss_index,
#         docstore=InMemoryDocstore(dict(enumerate(documents))),
#         index_to_docstore_id={i: i for i in range(len(documents))}
#     )

#     return jsonify({"message": "PDF processed and indexed successfully", "chunks": len(texts)})

# @app.route("/query", methods=["POST"])
# def query_pdf():
#     global index
#     if not index:
#         return jsonify({"error": "No PDF indexed yet"}), 400
    
#     try:
#         data = request.get_json()
#         if data is None or "query" not in data:
#             return jsonify({"error": "Query text is missing in the request."}), 400
        
#         query_text = data["query"]
#         query_embedding = embedding_model.embed_query(query_text)
        
#         distances, indices = index.index.search(np.array([query_embedding], dtype=np.float32), k=5)
#         retrieved_docs = [documents[i].page_content for i in indices[0]]

#         return jsonify({"query": query_text, "results": retrieved_docs})
    
#     except Exception as e:
#         return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import os
import re
import torch
import faiss
import numpy as np
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map={"": device},
    torch_dtype=torch.float16 if device == "cuda" else torch.bfloat16
).to(device)

index = None  # FAISS index

class TinyLlamaLLM(LLM):
    def _call(self, prompt: str, stop=None):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    @property
    def _llm_type(self):
        return "tinyllama"

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global index
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, file.filename)
    file.save(filepath)
    
    pdfreader = PdfReader(filepath)
    raw_text = "".join(page.extract_text() or "" for page in pdfreader.pages)
    clean_text = re.sub(r'[^\w\s]', '', raw_text)
    
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_text(clean_text)
    documents = [Document(page_content=text) for text in texts]
    
    embeddings = embedding_model.embed_documents(texts)
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    
    index = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=InMemoryDocstore(dict(enumerate(documents))),
        index_to_docstore_id={i: i for i in range(len(documents))}
    )
    
    return jsonify({"message": "PDF processed and indexed successfully", "chunks": len(texts)})

@app.route("/query", methods=["POST"])
def query_pdf():
    global index
    if not index:
        return jsonify({"error": "No PDF indexed yet"}), 400
    
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query text is missing"}), 400
    
    query_text = data["query"]
    query_embedding = embedding_model.embed_query(query_text)
    distances, indices = index.index.search(np.array([query_embedding], dtype=np.float32), k=5)
    
    retrieved_context = "\n".join([index.docstore._dict[i].page_content for i in indices[0]])
    
    qa_prompt = f"""
    Context:
    {retrieved_context}
    
    Question:
    {query_text}
    
    Provide a structured, concise response with bullet points without repetition.
    **Only answer the question directly without adding extra details or solutions.**
    """
    
    llm = TinyLlamaLLM()
    qa_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(qa_prompt))
    answer = qa_chain.invoke({"context": retrieved_context, "question": query_text})
    
    return jsonify({"response": answer["text"]})

if __name__ == '__main__':
    app.run(debug=True)
