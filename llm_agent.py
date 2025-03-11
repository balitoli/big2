# llm_agent.py
from openai import OpenAI
from config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_API_URL, OLLAMA_BASE_URL, OLLAMA_API_KEY, TEMPERATURE


if LLM_PROVIDER.upper() == "OPENAI":
    OPENAI_CHAT_CLIENT = OpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY)
elif LLM_PROVIDER.upper() == "OLLAMA":
    OPENAI_CHAT_CLIENT = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

#Advise Agent
def get_llm_advice(prompt, model):
    response = OPENAI_CHAT_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You're an expert Hong Kong Big Two (鋤大DEE) advisor."}, {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=TEMPERATURE
    )
    # The answer is typically in choices[0].message.content
    answer = response.choices[0].message.content.strip()
    return answer

#Opponent Players
