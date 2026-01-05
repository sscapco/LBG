import streamlit as st
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import hashlib
import re

# Import the governance pipeline 
from core.workflow import process_query_sync
from cortex_connection import cortex
from core.cortex_utils import cortex_chat_text


# Configure page
st.set_page_config(
    page_title="Governance Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for green theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2d5016;
        --secondary-green: #4a7c2c;
        --light-green: #7fb069;
        --pale-green: #e8f5e9;
        --accent-green: #90ee90;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d5016 0%, #4a7c2c 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: #7fb069 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem 0 !important;
        width: 100% !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: #90ee90 !important;
        transform: translateX(5px);
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f5f5f5;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    [data-testid="stChatMessageContent"] {
        background-color: transparent !important;
    }
    
    /* User message */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, #7fb069 0%, #90ee90 100%);
    }
    
    .stChatMessage[data-testid*="user"] p {
        color: white !important;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid*="assistant"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%);
        border-left: 4px solid #4a7c2c;
    }
    
    /* Input box */
    .stChatInput {
        border: 2px solid #7fb069 !important;
        border-radius: 12px !important;
    }
    
    /* Headers */
    h1 {
        color: #2d5016 !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: #4a7c2c !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #4a7c2c 0%, #7fb069 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
    }
    
    /* Conversation history items */
    .conversation-item {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 3px solid #90ee90;
    }
    
    .conversation-item:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    .conversation-item.active {
        background-color: rgba(144, 238, 144, 0.3);
        border-left: 3px solid #90ee90;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #4a7c2c !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def generate_conversation_id():
    """Generate unique conversation ID"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]


def generate_conversation_title(first_message: str) -> str:
    """Generate a short title summarizing the conversation"""
    def _fallback_title(msg: str) -> str:
        tokens = re.findall(r"[a-z0-9]+(?:[/-][a-z0-9]+)*", (msg or "").lower())
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "before", "can", "could", "do", "does", "for",
            "from", "get", "how", "i", "in", "is", "it", "me", "most", "need", "of", "on", "or",
            "our", "please", "should", "tell", "that", "the", "then", "this", "to", "us", "we",
            "what", "when", "where", "which", "who", "why", "you", "your",
        }
        kept = [t for t in tokens if t not in stop and len(t) >= 3]
        if not kept:
            return (msg or "Conversation")[:30] + ("..." if msg and len(msg) > 30 else "")
        # Prefer a small, specific phrase.
        phrase = kept[:4]
        phrase = [p.replace("/", " ").replace("-", " ") for p in phrase]
        title = " ".join(" ".join(phrase).split())
        return title[:40].strip()

    def _sanitize(title: str) -> str:
        t = (title or "").strip().strip('"').strip("'")
        t = re.sub(r"\s+", " ", t).strip()
        # Remove trailing punctuation / overly generic outputs.
        t = t.strip(" .,:;—-")
        return t

    generic = {"final", "security", "question", "help", "status", "launch", "approval", "approvals"}

    try:
        msg = (first_message or "").strip()
        if not msg:
            return "Conversation"

        # Prefer deterministic titles when possible (faster + more consistent than LLM).
        from core.governance_data import get_governance_data
        governance_data = get_governance_data()

        sid, method, conf = governance_data.deterministic_match(msg)
        if sid and conf >= 0.85:
            rec = governance_data.get_step_record(sid)
            if rec and rec.get("name"):
                return _sanitize(f"{sid}: {rec['name']}")[:50]

        # If this looks like name validation, label it directly.
        try:
            from core.automation_handler import detect_automation_request
            if detect_automation_request(msg):
                return "Name Validation"
        except Exception:
            pass

        prompt = f"""Generate a very brief title (3-5 words max) that summarizes this governance question:

"{first_message}"

Examples:
- "DPWG Presentation Status"
- "Next Steps After DPC"
- "DOI Form Progress"
- "Stage 0 Completion"

Return only the title, nothing else."""

        messages = [{"role": "user", "content": prompt}]
        response = cortex_chat_text(cortex.get_chat_response(
            messages,
            max_tokens=50,
            temperature=0.3,
            thinking_enabled=False
        ))

        title = _sanitize(response)
        if not title:
            return _fallback_title(first_message)
        if title.lower() in generic or len(title) < 4:
            return _fallback_title(first_message)
        return title
    except Exception:
        # Fallback to truncated message
        return _fallback_title(first_message)


def start_new_conversation():
    """Start a new conversation"""
    conversation_id = generate_conversation_id()
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = []
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": "New Conversation",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }


def load_conversation(conversation_id: str):
    """Load a conversation from history"""
    if conversation_id in st.session_state.conversations:
        st.session_state.current_conversation_id = conversation_id
        st.session_state.messages = st.session_state.conversations[conversation_id]["messages"].copy()


def save_current_conversation():
    """Save current conversation to history"""
    if st.session_state.current_conversation_id:
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages.copy()


# Sidebar - Conversation History
with st.sidebar:
    st.title("🌿 Governance Assistant")
    st.markdown("---")
    
    # New conversation button
    if st.button("➕ New Conversation", use_container_width=True):
        start_new_conversation()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📝 Conversation History")
    
    # Display conversations
    if st.session_state.conversations:
        # Sort by creation time (newest first)
        sorted_conversations = sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        
        for conv_id, conv_data in sorted_conversations:
            is_active = conv_id == st.session_state.current_conversation_id
            
            created_time = datetime.fromisoformat(conv_data["created_at"])
            title = conv_data.get("title") or "Conversation"
            button_label = f"{'🟢 ' if is_active else ''}  {title}"

            cols = st.columns([0.72, 0.28], gap="small")
            with cols[0]:
                if st.button(button_label, key=f"conv_{conv_id}", use_container_width=True):
                    load_conversation(conv_id)
                    st.rerun()
            with cols[1]:
                st.caption(created_time.strftime("%b %d, %I:%M %p"))
    else:
        st.info("No conversations yet. Start chatting to create one!")
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Governance Q&A Assistant**
        
        Ask questions about your workflow progress, next steps, and dependencies.
        
        **Example questions:**
        - "I completed the jira ticket, what's next?"
        - "I'm working on the DOI form"
        - "What comes after DPWG?"
        """)


# Main chat interface
st.title("🌿 Governance Workflow Assistant")
st.markdown("Ask me about your governance workflow progress, next steps, or any questions!")

# Initialize conversation if none exists
if not st.session_state.current_conversation_id:
    start_new_conversation()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your workflow progress..."):
    # Check if this is the first message in the conversation
    is_first_message = len(st.session_state.messages) == 0
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate title for first message
    if is_first_message:
        with st.spinner("Analyzing your question..."):
            title = generate_conversation_title(prompt)
            st.session_state.conversations[st.session_state.current_conversation_id]["title"] = title
    
    # Get response from governance pipeline
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your workflow..."):
            try:
                # Run the async function
                response = process_query_sync(
                    prompt,
                    session_id=st.session_state.current_conversation_id
                )
                                        
                # Display response
                st.markdown(response)
                
                # Add to messages
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"I encountered an error processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Save conversation
    save_current_conversation()
    
    # Force rerun to update sidebar
    st.rerun()
