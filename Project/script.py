from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader
import re
from langchain_community.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import shutil
from flask import Flask, jsonify, request

app = Flask(__name__)

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model = "gemini-pro", temperatue = 0.3 , max_retries=3)

current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_dir, 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

text_storage = []
embeddings = []
global_filepath = ""

def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\.', sep_token, s)
    return s

def allowedFile(filename):

    split_tup = os.path.splitext(filename)
    file_extension = split_tup[1]

    if file_extension == '.pdf':
        return True
    
    return False

@app.route('/api/upload' , methods=['POST'])
def getPDF():

    global global_filepath
    file = request.files['file']
    filename = file.filename
    
    if allowedFile(filename) == False:
        return jsonify({'error': "File not a pdf"}), 400
    
    if file.filename == "":
        return jsonify({'error': "No file selected"}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': "No file part"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'] , filename)
    file.save(filepath)
    ProcessPDF(filepath)
    global_filepath = filepath
    return jsonify({'response': "File Uploaded Successfully"}), 200

def ProcessPDF(filepath):
    doc_reader = PdfReader(filepath)
    raw_text = ""

    for i , page in enumerate(doc_reader.pages):
        text = page.extract_text()

        if text:
            raw_text = raw_text + text

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    texts = text_splitter.split_text(raw_text)
    texts = list(map(normalize_text , texts))
    print(type(texts))
    global text_storage
    text_storage = texts
    global embeddings
    embeddings = createVectorStore()
    

def createVectorStore():

    global text_storage
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_storage,embeddings)
    vector_store.save_local("faiss_index")

    if not os.path.exists("faiss_index"):
        return jsonify({'error': "Vector store did not get created"}), 400        

    return embeddings

def load_faiss_index_safely(index_path, embeddings):
    try:
        new_db = FAISS.load_local(index_path, embeddings , allow_dangerous_deserialization=True)
        return new_db
    except Exception as e:
        print(f"An error occurred during deserialization: {e}")
        raise

def get_conversation_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    global model

    prompt = PromptTemplate(template = prompt_template, input_variables= ["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)

    return chain

@app.route('/api/prompt' , methods = ['POST'])
def prompt():
    global embeddings

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    new_db = load_faiss_index_safely("faiss_index", embeddings)

    if new_db == None:
        return jsonify({'error': "Deserialization issue"}), 400 
    
    docs = new_db.similarity_search(prompt)
    chain = get_conversation_chain()

    response = chain.invoke({"input_documents":docs, "question": prompt})

    if response['output_text'] == None or response['output_text'] == "":
        return jsonify({'error': "Response is empty"}), 400

    print(response['output_text'])

    return jsonify({'response': response['output_text']}), 200

@app.route('/api/exit' , methods = ['GET'])
def exit_application():
    
    global global_filepath
    shutil.rmtree("faiss_index")
    os.remove(global_filepath)
    return jsonify({'response': "Exited successfully"}), 200

@app.route('/api/prompt/topic' , methods=['GET'])
def TellTopic():

    TOPIC = "This is a chatbot which is used to chat with Budget of India related questions"

    return jsonify({'response': TOPIC}), 200

if __name__ == '__main__':
    app.run(debug=True)