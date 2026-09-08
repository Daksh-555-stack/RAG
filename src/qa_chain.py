"""Retrieval + answer generation.

Builds a LangChain retrieval chain that:
  1. retrieves the top-k most relevant chunks from the FAISS index,
  2. feeds them — plus a strict grounding prompt — to an LLM,
  3. returns the answer together with the retrieved source documents.

The chain uses modern LCEL composition (``create_retrieval_chain`` from
``langchain.chains.retrieval``) instead of the legacy ``RetrievalQA`` class:
the legacy ``langchain.chains.base.Chain`` machinery crashes under Python
3.14 (a pydantic forward-ref evaluation bug), while the LCEL path runs on
every supported Python version.

The LLM is pluggable: free Hugging Face Inference API by default, with
OpenAI and Groq as optional providers (selected in the Streamlit sidebar).
A fully-local transformers alternative is included as a commented block.
"""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------------
# Prompt: forces the LLM to answer ONLY from the provided context and to say
# "I couldn't find this…" instead of hallucinating.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided document context.
If the answer is not contained in the context, say "I couldn't find this information in the uploaded document."
Do not use outside knowledge. Where possible, support your answer with short quotes from the context.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Default model for the free Hugging Face Inference API provider.
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# Lighter/faster HF alternatives: "google/flan-t5-large", "HuggingFaceH4/zephyr-7b-beta"

N_RETRIEVED_DOCS = 4  # how many chunks are fed to the LLM per question


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def build_llm(
    provider: str,
    api_key: str,
    hf_model: str = DEFAULT_HF_MODEL,
    openai_model: str = "gpt-4o-mini",
    groq_model: str = "llama-3.3-70b-versatile",
) -> BaseLanguageModel:
    """Return a LangChain LLM for the chosen provider.

    Args:
        provider: One of ``"huggingface"``, ``"openai"``, ``"groq"``
            (case-insensitive; the sidebar label is normalised here).
        api_key: API key for the provider (token for Hugging Face).
        hf_model: Hugging Face model id used by the free provider.
        openai_model: OpenAI chat model name.
        groq_model: Groq chat model name.

    Returns:
        A LangChain ``BaseLanguageModel`` instance.

    Raises:
        ValueError: If the provider is unknown or the API key is missing.
        RuntimeError: If the provider integration fails to initialise.
    """
    provider_key = provider.strip().lower()
    if "huggingface" in provider_key or "hugging face" in provider_key:
        return _build_huggingface(api_key, hf_model)
    if "openai" in provider_key:
        return _build_openai(api_key, openai_model)
    if "groq" in provider_key:
        return _build_groq(api_key, groq_model)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Choose from: "
        "Hugging Face (free), OpenAI, Groq."
    )


def _build_huggingface(api_key: str, model: str) -> BaseLanguageModel:
    """Free Hugging Face Inference API endpoint (needs a free HF token)."""
    # Deferred import keeps startup fast and pins the failure to this call.
    from langchain_huggingface import HuggingFaceEndpoint

    if not api_key:
        raise ValueError(
            "No Hugging Face token found. Add HUGGINGFACEHUB_API_TOKEN to your "
            ".env file or paste a token in the sidebar "
            "(https://huggingface.co/settings/tokens)."
        )
    try:
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise the Hugging Face endpoint: {exc}") from exc


def _build_openai(api_key: str, model: str) -> BaseLanguageModel:
    """OpenAI chat model (requires an OpenAI API key)."""
    from langchain_openai import ChatOpenAI

    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Add OPENAI_API_KEY to your .env file or "
            "paste a key in the sidebar."
        )
    try:
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise OpenAI: {exc}") from exc


def _build_groq(api_key: str, model: str) -> BaseLanguageModel:
    """Groq chat model (requires a Groq API key)."""
    from langchain_groq import ChatGroq

    if not api_key:
        raise ValueError(
            "No Groq API key found. Add GROQ_API_KEY to your .env file or "
            "paste a key in the sidebar."
        )
    try:
        return ChatGroq(model=model, api_key=api_key, temperature=0.2)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Groq: {exc}") from exc


# ---------------------------------------------------------------------------
# Retrieval chain
# ---------------------------------------------------------------------------
def _format_docs(documents: list) -> str:
    """Join retrieved chunks into the prompt's ``{context}`` block."""
    return "\n\n".join(doc.page_content for doc in documents)


def create_qa_chain(vector_store, llm: BaseLanguageModel, k: int = N_RETRIEVED_DOCS):
    """Build a retrieval chain over a FAISS vector store (LCEL).

    Args:
        vector_store: A FAISS vector store (see ``vector_store.create_vector_store``).
        llm: A LangChain LLM (see ``build_llm``).
        k: Number of chunks to retrieve per question.

    Returns:
        An LCEL ``Runnable``. Invoke it with ``{"input": question}``; the result
        dict contains:
          - ``answer``: the grounded answer string,
          - ``context``: the list of retrieved source ``Document`` objects
            (the equivalent of RetrievalQA's ``return_source_documents=True``).
    """
    from langchain.chains.retrieval import create_retrieval_chain

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # The combine step receives {"input": ..., "context": [documents]}.
    # Map "input" → "question" for our template, then run prompt → LLM → text.
    combine_docs_chain = (
        RunnablePassthrough.assign(question=lambda inputs: inputs["input"])
        | PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        return create_retrieval_chain(retriever, combine_docs_chain)
    except Exception as exc:
        raise RuntimeError(f"Failed to create the QA chain: {exc}") from exc


# ---------------------------------------------------------------------------
# Fully-local alternative (zero API calls, no token needed)
# ---------------------------------------------------------------------------
# If you want to remove the API dependency entirely, swap ``build_llm`` for a
# local transformers pipeline. Note the model must be downloaded once and that
# a 7B model needs several GB of RAM:
#
#     from langchain_huggingface import HuggingFacePipeline
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
#     model_name = "google/flan-t5-large"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#     llm = HuggingFacePipeline(pipeline=pipe)
#
# Then pass ``llm`` straight into ``create_qa_chain``.
