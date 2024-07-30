from sentence_transformers import SentenceTransformer, util
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ['API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

@app.route('/api/prompt/topic' , methods=['GET'])
def TellTopic():

    TOPIC = "This is a chatbot which is used to chat with Budget of India related questions"

    return jsonify({'response': TOPIC}), 200

def CheckPrompt(prompt):

    topic = "Budget and tax policies in India"
    prompt_embedding = sentence_transformers_model.encode(prompt, convert_to_tensor=True)
    topic_embedding = sentence_transformers_model.encode(topic, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(prompt_embedding, topic_embedding)
    similarity1 = similarity.item() > 0.7 
    return  similarity1

@app.route('/api/prompt' , methods=['POST'])
def getPrompt():

    data = request.get_json()
    prompt = data.get('prompt')
    bval = CheckPrompt(prompt)

    if bval == False:
        return jsonify({'error': 'Prompt Not related to Topic'}), 400

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        response = model.generate_content(prompt)
        
        candidates = response.candidates
        if not candidates:
            return jsonify({'error': 'No content generated'}), 500

        response_text = candidates[0].content.parts[0].text
        return jsonify({'response': response_text}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)