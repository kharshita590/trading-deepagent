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
git clone https://github.com/kharshita590/trading-deepagent.git
cd src
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

## 📄 License

MIT License
