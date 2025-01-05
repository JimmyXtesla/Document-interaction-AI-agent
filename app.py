import os
from flask import Flask, request, jsonify, render_template
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PERSIST_DIRECTORY = "chromadb_persist"
COLLECTION_NAME = "my_research_papers"
OLLAMA_MODEL = "llama2"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize ChromaDB
def create_or_load_collection(persist_directory, collection_name):
    client = chromadb.PersistentClient(path=persist_directory)

    if collection_name in client.list_collections():
      collection = client.get_collection(collection_name)
      print(f"Collection '{collection_name}' already exists. Using existing collection.")
    else:
      print(f"Creating new collection: '{collection_name}'")
      openai_ef = embedding_functions.SentenceTransformerEmbeddingFunction(api_key="empty", model_name="all-MiniLM-L6-v2")
      collection = client.create_collection(name=collection_name, embedding_function=openai_ef)

    return collection

collection = create_or_load_collection(PERSIST_DIRECTORY, COLLECTION_NAME)

# Function to load documents based on file extension
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# Function to process and store documents
def process_and_store_documents(file_paths):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_path in file_paths:
        docs = load_document(file_path)
        chunks = text_splitter.split_documents(docs)
        ids = [f"{file_path}_{i}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {"source": file_path, "chunk": i, 'page': chunk.metadata.get('page', 'unknown')} for i, chunk in enumerate(chunks)
        ]
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
    
    print(f"Added {len(file_paths)} files to the vectorstore.")

# Function to setup RAG system for query answering
def setup_rag_chain():
    llm = Ollama(model=OLLAMA_MODEL)
    retriever = collection.as_retriever()
    prompt_template = """
        You are an expert assistant that helps with research paper reading. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

    return qa
qa_chain = setup_rag_chain()


# Function to generate a literature review
def generate_literature_review(topic, num_papers=5):
    llm = Ollama(model=OLLAMA_MODEL)
    retriever = collection.as_retriever(search_kwargs={"k": num_papers})
    relevant_docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
        You are an expert researcher. Based on the following documents, write a literature review on the topic "{topic}". 
        
        The literature review should include:
        - A brief introduction to the topic
        - Overview of the main findings of the documents.
        - Any gaps in research.
        - Any open questions.
        
        Documents:
        {context}

        Literature Review:
    """

    review = llm(prompt)
    return review


# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400

      files = request.files.getlist('files')
      uploaded_files = []
      
      for file in files:
          if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)

      if uploaded_files:
        process_and_store_documents(uploaded_files)
        return jsonify({"message": "Documents uploaded and processed"}), 200
      else:
        return jsonify({"message": "No files selected or invalid file extension."}), 400
      
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_type = data.get('query_type')
    query_text = data.get('query_text')
    
    if not query_text:
      return jsonify({'error': "Query text cannot be empty."}), 400

    if query_type == 'question':
        answer = qa_chain.run(query_text)
        return jsonify({'answer': answer}), 200
    elif query_type == 'review':
        literature_review = generate_literature_review(query_text)
        return jsonify({'review': literature_review}), 200
    else:
      return jsonify({"error": "Invalid query type"}), 400

if __name__ == '__main__':
  os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist
  app.run(debug=True, host="0.0.0.0", port=5000)