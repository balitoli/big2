##CONFIG FILE
from dotenv import load_dotenv
import os

load_dotenv()
openai_apikey = os.getenv('YOUR_OPENAI_API_KEY')
openai_url = os.getenv('OPENAI_BASE_URL')

# Choose providers for each functionality:
LLM_PROVIDER = "OPENAI"         # Options: "OPENAI", "OLLAMA"

# For OpenAI (applies to both LLM and embeddings if selected)
OPENAI_API_KEY = openai_apikey
OPENAI_API_URL = openai_url

# For Ollama “Shim” client info
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Example: adjust to your local server
OLLAMA_API_KEY = "ollama"  # placeholder but required by the client

#Which Model To Use
OPENAI_LLM_MODEL = "deepseek-r1"             # OpenAI (gpt-4o; gpt-4o-mini; o1-mini; o3-mini-all; etc. [deepseek-r1; grok3; etc.] if you have a third-party universal API for testing)
OLLAMA_LLM_MODEL = "llama3.2"         # Ollama (llama3.2; gemma2:9b; gemma2:27b; phi4; dolphin-mixtral:8x7b; dolphin-mistral; mistral; deepseek-r1:8b etc.)

#Temperature of the LLM model output
TEMPERATURE = 0

# === Configuration & UI Constants ===
WIDTH, HEIGHT = 2600, 1900