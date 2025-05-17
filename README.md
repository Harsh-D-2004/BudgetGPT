# 💬 BudgetGPT

**BudgetGPT** is a chatbot application designed to handle queries related to the **Budget of India** using a PDF document as its source. It processes the document, creates vector embeddings, and enables interactive question-answering through a RESTful API.

---

## 🚀 Features

- 📄 Upload PDF documents
- 📘 Text extraction from PDFs
- 🤖 Generate embeddings using Google Generative AI
- 💬 Chatbot-based question answering
- 🌐 RESTful API endpoints for interaction

---

## 🛠️ Tech Stack

- **Backend:** Flask
- **Embeddings:** Google Generative AI
- **Text Processing:** `pypdf`, `langchain`
- **Vector Store:** FAISS
- **Environment Management:** `python-dotenv`

---

## 📁 Directory Structure

├── .env
├── .gitattributes
├── Procfile
├── README.md
├── requirements.txt
├── script.py
├── faiss_index
│ ├── index.faiss
│ └── index.pkl
└── upload
└── BudgetGPT_PDF.pdf

## 📦 Prerequisites

- Python 3.x
- A .env file with the GOOGLE_API_KEY
- Libraries specified in requirements.txt:
- Flask
- langchain
- pypdf

## Backend Setup
- pip install -r requirements.txt
- python script.py

- Port Information: The Flask application runs on the default port 5000.

- API Endpoints:

- /api/upload: Upload PDF files for processing.
- /api/prompt: Send a prompt to the chatbot.
- /api/exit: Clean up resources and exit.
- /api/prompt/topic: Get information about the budget topic functionality.

## Frontend Setup

- npm install
- npm run dev

## Screenshots
