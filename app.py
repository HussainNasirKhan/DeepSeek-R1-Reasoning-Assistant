import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main content and general styling */
    .main {
        background-color: #f5f5f5;
        color: #333333;
    }
    
    /* Sidebar width control */
    [data-testid="stSidebar"] {
        width: 300px !important;
    }
    
    [data-testid="stSidebarNav"] {
        width: 300px !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        width: 300px !important;
    }
    
    /* Select box styling */
    .stTextInput textarea {
        color: #333333 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox svg {
        fill: #333333 !important;
    }
    
    .stSelectbox option {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    div[role="listbox"] div {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    /* Make sidebar content more compact */
    .sidebar .sidebar-content {
        padding: 1rem 0.5rem !important;
    }
    
    /* Adjust sidebar header margins */
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤔 DeepSeek-R1: Reasoning Assistant")
st.caption("💡 Your AI Partner for Critical Thinking and Analysis")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:7b", "deepseek-r1:1.5b"],
        index=0
    )
    
    reasoning_type = st.selectbox(
        "Reasoning Approach",
        ["Analytical", "Critical", "Strategic", "Systems Thinking"],
        index=0
    )

# Initialize the LLM engine
llm_engine = Ollama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.7
)

# Updated system prompt for tagged responses
system_template = """You are a concise reasoning assistant specializing in {reasoning_type} thinking. 
Always structure your response in TWO parts using XML tags:

<think>
Here, provide your step-by-step analysis:
- Break down the key aspects of the problem
- Consider important factors and relationships
- Identify any assumptions or constraints
Be thorough but concise in your thinking process.
</think>

<response>
Here, provide your clear, actionable conclusion:
- Present your final analysis
- Keep it under 150 words
- Use simple, direct language
- Focus on practical insights
</response>

Always enclose your thinking in <think></think> tags and your final response in <response></response> tags."""

system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{
        "role": "ai", 
        "content": "Hello! I'm your reasoning assistant. Share your thoughts or questions, and I'll help you analyze them systematically. 🤔",
        "is_initial": True
    }]

# Chat container
chat_container = st.container()

# Display chat messages with proper tag parsing
with chat_container:
    for message in st.session_state.message_log:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("ai"):
                if message.get("is_initial", False):
                    # Display initial message without tabs
                    st.markdown(message["content"])
                else:
                    content = message["content"]
                    
                    # Extract content between tags
                    think_content = ""
                    if "<think>" in content and "</think>" in content:
                        think_content = content.split("<think>")[1].split("</think>")[0].strip()
                    
                    response_content = ""
                    if "<response>" in content and "</response>" in content:
                        response_content = content.split("<response>")[1].split("</response>")[0].strip()
                    
                    # Create tabs for non-initial messages
                    tab1, tab2 = st.tabs(["💭 Thinking", "✨ Response"])
                    with tab1:
                        st.markdown(think_content)
                    with tab2:
                        st.markdown(response_content)

# Chat input and processing
user_query = st.chat_input("Share your thoughts or questions...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({"reasoning_type": reasoning_type})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai" and not msg.get("is_initial", False):
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🤔 Analyzing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log (not initial message)
    st.session_state.message_log.append({
        "role": "ai",
        "content": ai_response,
        "is_initial": False
    })
    
    # Rerun to update chat display
    st.rerun()