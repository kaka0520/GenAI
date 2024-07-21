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
def get_llm(model_name: str, temperature: float, top_p: float):
    print (f'In get_llm function.. ')
    print (f'temperature: {temperature}')
    print (f'top_p: {top_p}')
    if model_name == "qwen2-7b":
        st.session_state.models[model_name] = Ollama(
            model="qwen2:7b",
            base_url="http://localhost:11434",
            verbose=True,
            temperature=temperature,
            top_p=top_p
        )
    elif model_name == "Llama3-8B":
        st.session_state.models[model_name] = Ollama(
            model="llama3",
            base_url="http://localhost:11434",
            verbose=True,
            temperature=temperature,
            top_p=top_p
        )
    else:
        st.session_state.models[model_name] = Ollama(
            model="llama3",
            base_url="http://localhost:11434",
            verbose=True,
            temperature=temperature,
            top_p=top_p
        )
    
    # Update the current_model in session state
    st.session_state.current_model = model_name

    print (f'current_model name: ')
    print (st.session_state.current_model)
    print (f'model object: ')
    print (st.session_state.models[model_name])
    
    # Return the model instance
    return st.session_state.models[model_name]

def clear_chat_history():
    st.session_state.messages = \
            [Message(
                actor=ASSISTANT,
                payload="Hello! I am here, ask me your questions")]
# def get_llm_chain(prompt_context, model_name, temperature, top_p, max_length):
#     if prompt_context is None or prompt_context.strip() == "":
#         prompt_context = "You are deeply knowledgeable about any questions."
#     template = """Answer the question based on the context below and use the history of the conversation to continue.
                              
#                 Context:
#                 """ + prompt_context + """

#                 History of the conversation:
#                 {chat_history}
                
#                 Question: 
#                 {question}

#                 Answer: 
#     """
#     prompt_template = PromptTemplate.from_template(template)
#     # Notice that we need to align the `memory_key`
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     chain = LLMChain(
#         llm=get_llm(model_name, temperature, top_p, max_length),
#         prompt=prompt_template,
#         verbose=True,
#         memory=memory
#     )
#     return chain


USER = "user"
ASSISTANT = "assistant"

def initialize_session_state():
    st.set_page_config(page_title="Kaka Chat with GenAI", page_icon=":robot_face:")

    # st.file_uploader(
    #     "Upload your files",
    #     type=["pdf", "txt"],
    #     key="file_uploader",
    #     on_change=read_and_save_file,
    #     label_visibility="collapsed",
    #     help="Upload your files here",
    #     accept_multiple_files=True,
    # )
    # st.session_state["ingestion_spinner"] = st.empty()

    if "messages" not in st.session_state:
        st.session_state["messages"] = \
            [Message(
                actor=ASSISTANT,
                payload="Hello! I am here, ask me your questions")]

    if 'models' not in st.session_state:
        st.session_state["models"] = {}

    # if "llm_chain" not in st.session_state:
        # st.session_state["llm_chain"] = get_llm_chain(None, "Llama3-8B")

    if 'prev_temperature' not in st.session_state:
        st.session_state.prev_temperature = 0.1  # Default value
    if 'prev_top_p' not in st.session_state:
        st.session_state.prev_top_p = 0.9  # Default value

    with st.sidebar:
        st.title("💬 Kaka is dancing with GenAI")
        st.caption("🚀 A chatbot with langchain + llama3 + Streamlit")
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a LLM model', ['Llama3-8B', 'qwen2-7b', 'Gemini-1.5'], key='selected_model')
        # Display the selected model on the main page
        st.write(f"You have selected: {selected_model}")

        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=st.session_state.prev_temperature, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=st.session_state.prev_top_p, step=0.01)
        # max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

        current_model = get_llm(selected_model, temperature, top_p)
        print (f'current_model:  {current_model}')
        if temperature != st.session_state.prev_temperature or top_p != st.session_state.prev_top_p:
            st.session_state.current_model = current_model
            st.session_state.prev_temperature = temperature
            st.session_state.prev_top_p = top_p

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def get_llm_chain_from_session(model_name) -> LLMChain:
    return st.session_state.models[model_name]
    # return st.session_state.get(model_name, None)
    # return st.session_state["models"]

# def read_and_summarize_file(file):
#     # Read the file content and prepare it as a prompt
#     content = file.read().decode()
#     prompt = f"Summarize for what you learn from the following content:\n{content}"
#     return prompt

# def read_and_save_file():
#     # st.session_state["assistant"].clear()
#     # st.session_state["messages"] = []
#     st.session_state["user_input"] = ""
#     st.session_state.messages.append(Message(actor=USER, payload="I have uploaded the file, please study from the content."))

#     for file in st.session_state["file_uploader"]:
#         prompt = read_and_summarize_file(file)
#         with st.session_state["ingestion_spinner"], st.spinner(f"Loading file {file.name}"):
#             response = get_llm_chain(prompt)
#             llm_chain = get_llm_chain_from_session()
#             response: str = llm_chain({"question": prompt})["text"]
#             st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response))
#             st.chat_message(ASSISTANT).write(response)
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(file.getbuffer())
#             file_path = tmp_file.name
#         os.remove(file_path)


initialize_session_state()

msg: Message
for msg in st.session_state["messages"]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("How can I help you?")


if prompt:
    st.session_state["messages"].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait.."):
        selected_model = st.session_state.selected_model  
        print (f'selected_model: {selected_model}')
        llm_chain = get_llm_chain_from_session(selected_model)
        response: str = llm_chain.predict(prompt)
        st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)