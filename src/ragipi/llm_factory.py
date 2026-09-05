"""
llm_factory.py — Fabrique de LLMs multi-provider
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Providers supportés :
  - openai   : GPT-3.5-turbo (commercial, aligné RLHF)
  - groq     : LLaMA 3.3 70B (open source, via Groq)
  - deepseek : DeepSeek-V3 (open source, 5M tokens gratuits)
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str = "openai"):
    """
    Retourne une instance LLM LangChain selon le provider.
    Tous les providers partagent la même interface LangChain.
    """

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        )

    elif provider == "deepseek":
        # DeepSeek est compatible OpenAI — on utilise ChatOpenAI
        # avec une base_url différente
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
        )

    elif provider == "nvidia":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0,
        )

    else:
        raise ValueError(
            f"Provider inconnu : '{provider}'. "
            "Utilise 'openai', 'groq', 'deepseek' ou 'nvidia'."
        )


if __name__ == "__main__":
    for provider in ["openai", "groq", "deepseek", "nvidia"]:
        print(f"\nTest {provider}...")
        try:
            llm = get_llm(provider)
            response = llm.invoke("Reply with OK only.")
            print(f"  {provider} : {response.content.strip()}")
        except Exception as e:
            print(f"  {provider} ERREUR : {e}")
