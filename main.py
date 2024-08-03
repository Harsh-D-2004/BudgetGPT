from uuid import uuid4
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import FAISS
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import spacy
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.environ['API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")
UPLOAD_FOLDER = '/BudgetGPT/upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pdf_index = None


@app.route('/api/prompt/topic' , methods=['GET'])
def TellTopic():

    TOPIC = "This is a chatbot which is used to chat with Budget of India related questions"

    return jsonify({'response': TOPIC}), 200

def extractive_summary(content, query):
    doc = nlp(content)
    sentences = [sent.text for sent in doc.sents]

    query_embedding = sentence_transformers_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = sentence_transformers_model.encode(sentences, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings).squeeze().tolist()

    sorted_sentences = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)

    top_sentences = [sent for sent, score in sorted_sentences[:5]] 

    return ' '.join(top_sentences)

@app.route('/api/prompt' , methods=['POST'])
def getPrompt():
    global pdf_index
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        retriever = pdf_index.as_retriever(search_kwargs={"k": 2})

        docs = retriever.get_relevant_documents(prompt)

        scored_docs = []
        for doc in docs:
            doc_embedding = sentence_transformers_model.encode(doc.page_content, convert_to_tensor=True)
            query_embedding = sentence_transformers_model.encode(prompt, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            scored_docs.append((doc, similarity))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:3]:
            summarized_content = extractive_summary(doc.page_content, prompt)
            results.append({
                "content": summarized_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def allowedFile(filename):

    split_tup = os.path.splitext(filename)
    file_extension = split_tup[1]

    if file_extension == '.pdf':
        return True
    
    return False
    
@app.route('/api/upload' , methods=['POST'])
def uploadPDF():
    global pdf_index
    file = request.files['file']

    if 'file' not in request.files:
        return jsonify({'error' : 'File not uploaded 1'}) , 400
    
    if file.filename == '':
        return jsonify({'error' : 'File not uploaded 2'}) , 400
    
    bVal = allowedFile(file.filename)

    if bVal == False:
        return jsonify({'error' : 'File type not supported'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        model = SentenceTransformer('all-MiniLM-L6-v2') 

        texts = [chunk.page_content for chunk in chunks]
        embeddings = model.encode(texts)

        docs = [Document(page_content=text, metadata={"source": filepath}) for text in texts]

        docstore = InMemoryDocstore({str(uuid4()): doc for doc in docs})

        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=model.encode,
            docstore=docstore
        )

        pdf_index = vectorstore
        print(pdf_index)
    
        return jsonify({'response': "File Uploaded Successfully"}), 200
    
    return jsonify({'error' : 'Error Occured'}) , 400


if __name__ == '__main__':
    app.run(debug=True)