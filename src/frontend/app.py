import streamlit as st
from retrieval import VectorDBBuilder
from model import LLMModel
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def submit_function():
    if user_prompt:
        # Store user message in session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # Query the vector DB for context
        response = query_engine.query(user_prompt)

        if response.source_nodes == []:
            response_text = "Sorry, I can only answer questions based on the books for your grade."
            # Store assistant message in session state
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            # Display assistant response
            st.markdown(f"**🤖 Assistant:** {response_text}")
        else:
            # reformat response
            context = "Context:\n"
            for i in range(config['query_config']['top_k']):
                try:
                    context = context + response.source_nodes[i].text + "\n\n"
                except:
                    continue
            print(context)
            # Combine context with user prompt
            full_prompt = f"StudentGPT, a chatbot that answers students' questions based on their grade and the \
                            relevant books.communicates in clear, easy language, answer is short and brief. \
                            Context:\n \
                            {context}\n \
                            Please respond to the following question. Use the context above if it is helpful. \
                            if not helpful please respond with \"Sorry, I can only answer questions based on the books for your grade.\" \
                            \nUser Prompt:\n{user_prompt}"

            # Generate response from the LLM
            try:
                response_text = st.session_state.model.generate_response(
                    model=st.session_state.selected_model,
                    prompt=full_prompt,
                    max_tokens=512,
                )

                # Store assistant message in session state
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Display assistant response
                st.markdown(f"**🤖 Assistant:** {response_text}")
            except Exception as e:
                st.error(f"Error generating response: {e}")

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
query_engine = builder.load_vectordb(selected_grade)

# Display chat messages
for message in st.session_state.messages:
    role = "🤖 Assistant" if message["role"] == "assistant" else "👨‍💻 User"
    st.markdown(f"**{role}:** {message['content']}")

# Add spacing for better alignment
st.markdown("<div style='height: 30vh;'></div>", unsafe_allow_html=True)

# Input box for user prompt
with st.form("my_form"):
    col1, col2 = st.columns([5, 1])  # Create two columns: 5 parts for input, 1 part for button
    with col1:
        user_prompt = st.text_input("Enter your prompt here:")
    with col2:
        st.write("")  # Add an empty line to move the button down
        st.write("")  # Add an empty line to move the button down
        submitted = st.form_submit_button("➤")

if submitted or st.session_state.get("submit", False):
    submit_function()
    st.session_state.submit = False

st.markdown("""
<script>
    document.addEventListener("DOMContentLoaded", function() {
        const inputField = document.querySelector("input[type='text']");
        inputField.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                event.preventDefault();
                document.querySelector("form").submit();
                document.querySelector("input[type='text']").blur();
            }
        });
    });
</script>
""", unsafe_allow_html=True)