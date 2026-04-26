# trading-deepagent

Architected a multi-agent orchestration layer (DeepAgents) with LangGraph to run parallel workflows across Data,
Forecasting, Sentiment, Risk, and Execution agents to get final stock list to invest according to the needs.

---

---

## ⚙️ Prerequisites

* Python 3.9 or higher
* pip (Python package manager)
* OpenAI API key (if using OpenAI models)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/trading-deepagent.git
cd trading-deepagent
```

---

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain langchain-openai chromadb tiktoken python-dotenv
```

---

## 🔑 Environment Variables

Set your API key before running the project:

### Linux / Mac

```bash
export OPENAI_API_KEY=your_api_key
```

### Windows

```bash
set OPENAI_API_KEY=your_api_key
```

---

## 🚀 Running the Application

Navigate to the agent directory:

```bash
cd app/agents
```

Run the main script:

```bash
python main.py
```

---

## 💬 Entering User Query

Once the program starts, you will be prompted to enter a query in the terminal.

Example:

```
Enter your trading query (or 'exit'): What is RSI?
```

To exit the application:

```
exit
```

---

## 📊 How It Works (Simple RAG Flow)

1. User enters a query
2. Query is converted into embeddings
3. Vector database retrieves relevant chunks
4. Retrieved context is passed to the model
5. Model generates a response

---

## 📌 Notes

* Ensure your `data/` folder contains trading-related text files
* The `vector_db/` directory will be created automatically on first run
* Internet connection is required if using external APIs

---

## 🧪 Future Improvements

* Add CLI arguments support
* Integrate real-time trading APIs
* Improve retrieval with hybrid search
* Add logging and evaluation metrics

---

## 📄 License

MIT License
