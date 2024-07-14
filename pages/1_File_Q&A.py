from dataclasses import dataclass
from langchain_community.llms import Ollama
import streamlit as st
import tempfile
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


@dataclass
class Message:
    actor: str
    payload: str


@st.cache_resource
def get_llm():
    return Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

def clear_chat_history():
    st.session_state.messages = \
            [Message(
                actor=ASSISTANT,
                payload="Hello! I am here, ask me your questions.")]
def get_llm_chain(prompt_context):
    if prompt_context is None or prompt_context.strip() == "":
        prompt_context = "You are deeply knowledgeable about any questions, includes documentation summary."
    template = """Answer the question based on the context below and use the history of the conversation to continue
                If the question cannot be answered using the information provided answer with "I don't know"
                              
                Context:
                """ + prompt_context + """

                History of the conversation:
                {chat_history}
                
                Question: 
                {question}

                Answer: 
    """
    prompt_template = PromptTemplate.from_template(template)
    # Notice that we need to align the `memory_key`
    memory = ConversationBufferMemory(memory_key="chat_history")
    chain = LLMChain(
        llm=get_llm(),
        prompt=prompt_template,
        verbose=True,
        memory=memory
    )
    return chain


USER = "user"
ASSISTANT = "assistant"

def initialize_session_state():
    st.set_page_config(page_title="Kaka Chat with Llama", page_icon=":robot_face:")

    st.file_uploader(
        "Upload your files",
        type=["pdf", "txt"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        help="Upload your files here",
        accept_multiple_files=True,
    )
    st.session_state["ingestion_spinner"] = st.empty()

    if "messages" not in st.session_state:
        st.session_state["messages"] = \
            [Message(
                actor=ASSISTANT,
                payload="Hello! I am here, ask me your questions. I will answer based on the content of file.")]

    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain(None)


    with st.sidebar:
        st.title("💬 Kaka is dancing with llama3")
        st.caption("🚀 A chatbot with langchain + llama3 + Streamlit")
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a LLM model', ['Llama3-8B', 'Llama3-70B', 'Gemini-1.5'], key='selected_model')
        if selected_model == 'Llama3-8B':
            llm = get_llm()
        elif selected_model == 'Llama3-70B':
            llm = 'test-Llama3-70B'
        elif selected_model == 'Gemini-1.5':
            llm = 'test-Gemini-1.5'
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]

def read_and_summarize_file(file):
    # Read the file content and prepare it as a prompt
    content = file.read().decode()
    prompt = f"Summarize for what you learn from the following content:\n{content}"
    return prompt

def read_and_save_file():
    # st.session_state["assistant"].clear()
    # st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state.messages.append(Message(actor=USER, payload="I have uploaded the file, please study from the content."))

    for file in st.session_state["file_uploader"]:
        prompt = read_and_summarize_file(file)
        with st.session_state["ingestion_spinner"], st.spinner(f"Loading file {file.name}"):
            response = get_llm_chain(prompt)
            llm_chain = get_llm_chain_from_session()
            response: str = llm_chain({"question": prompt})["text"]
            st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getbuffer())
            file_path = tmp_file.name
        os.remove(file_path)


initialize_session_state()

msg: Message
for msg in st.session_state["messages"]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("How can I help you?")


if prompt:
    st.session_state["messages"].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait.."):
        llm_chain = get_llm_chain_from_session()
        response: str = llm_chain({"question": prompt})["text"]
        st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)