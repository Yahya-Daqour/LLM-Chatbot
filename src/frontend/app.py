import streamlit as st
from retrieval import VectorDBBuilder
from model import LLMModel
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config_file = '../config/config.yaml'
config = load_config(config_file)

# Streamlit UI setup
st.set_page_config(page_icon="💬", layout="wide", page_title="RAG Chatbot")

# Cache initialization functions
@st.cache_resource
def load_vectordb_builder():
    return VectorDBBuilder(config_file='../config/config.yaml')

# Initialize components
builder = load_vectordb_builder()

# Session state for the model
if "model" not in st.session_state:
    st.session_state.model = LLMModel(api_key=st.secrets["groq_api_key"])

# Sidebar
st.sidebar.title("Settings")
grades = config['grades']
selected_grade = st.sidebar.selectbox("Choose a grade:", grades)

# Available LLM models
models = {
    "llama3-70b-8192": {
        "name": "LLaMA3-70b-Instruct",
        "tokens": 8192,
        "developer": "Meta",
    },
    "llama3-8b-8192": {
        "name": "LLaMA3-8b-Instruct",
        "tokens": 8192,
        "developer": "Meta",
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral-8x7b-Instruct-v0.1",
        "tokens": 32768,
        "developer": "Mistral",
    },
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
}

# Model selection and max tokens slider
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
)

# Initialize session state for messages and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

# if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_option

# Load the vector DB
index = builder.load_vectordb(selected_grade)

# Display chat messages
for message in st.session_state.messages:
    role = "🤖 Assistant" if message["role"] == "assistant" else "👨‍💻 User"
    st.markdown(f"**{role}:** {message['content']}")

# Input box for user prompt
user_prompt = st.text_input("Enter your prompt here:")

if user_prompt:
    # Store user message in session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Query the vector DB for context
    context = index.as_query_engine().query(user_prompt)

    # Check if context is a Response object, and extract the text properly
    if hasattr(context, 'results'):
        context_text = "\n".join([doc.text for doc in context.results])
    else:
        context_text = ""

    # Combine context with user prompt
    full_prompt = f"Context:\n{context_text}\n\nUser Prompt:\n{user_prompt}"

    # Generate response from the LLM
    try:
        response = st.session_state.model.generate_response(
            model=st.session_state.selected_model,
            prompt=full_prompt,
            max_tokens=512,
        )
        # Store assistant message in session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        st.markdown(f"**🤖 Assistant:** {response}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
