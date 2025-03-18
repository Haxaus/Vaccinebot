import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv  # For local development

# Load environment variables locally (optional)
load_dotenv()

DB_FAISS_PATH = "vector_db/"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2' )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

CUSTOM_PROMPT_TEMPLATE = """
You are a knowledgeable assistant specializing in vaccines in India. Below is the context from official vaccine data, followed by the user's question. Use the context to provide a clear, concise, and conversational answer that directly addresses the question’s intent. If the question asks for a count, provide only the number or a brief summary. If it asks for a list or details, provide those specifically without repeating unnecessary information. If the context doesn’t fully answer the question, say so honestly and avoid guessing.

Context: {context}
Question: {question}

Answer naturally, as if explaining to someone curious about vaccines in India.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def main():
    st.title("Vaccine Assistant 🇮🇳💉")
    st.write("Ask me about vaccines in India!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        # Fetch token securely: Streamlit Cloud secrets or local env
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Hugging Face token not found. Please set HF_TOKEN in secrets or environment.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE) }
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

   