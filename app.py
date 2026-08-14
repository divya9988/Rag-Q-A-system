import os
import tempfile

import streamlit as st

import config
from rag_pipeline import ingest_and_index, answer_query

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#eef7ff,#f8f1ff);
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Sidebar */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#4f46e5,#60a5fa);
    color:white;
}

/* Title */
.main-title{
    font-size:42px;
    font-weight:700;
    color:#4f46e5;
    text-align:center;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#666;
    margin-bottom:30px;
}

/* Upload Card */
.upload-card{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 10px 25px rgba(0,0,0,.08);
}

/* Chat User */
.user-msg{
    background:#60a5fa !important;
    color:white !important;
    padding:15px;
    border-radius:18px;
    margin-top:15px;
    margin-left:25%;
    box-shadow:0 8px 18px rgba(0,0,0,.08);
}

.user-msg,
.user-msg * {
    color: white !important;
}

/* Assistant */
.bot-msg{
    background:white !important;
    color:black !important;
    padding:18px;
    border-radius:18px;
    margin-right:25%;
    margin-top:15px;
    box-shadow:0 10px 25px rgba(0,0,0,.08);
}

.bot-msg,
.bot-msg * {
    color: black !important;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#4f46e5,#60a5fa);
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 25px;
    font-weight:bold;
}

/* Text Input */
.stTextInput>div>div>input{
    border-radius:15px;
    border:2px solid #d6d6ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# SESSION
# ---------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# HEADER
# ---------------------------

st.markdown(
    "<div class='main-title'>🤖 Intelligent RAG Assistant</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>Upload a document and chat with your AI assistant</div>",
    unsafe_allow_html=True,
)

# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    st.title("📂 Document")

    uploaded_file = st.file_uploader(
        "Upload",
        type=config.SUPPORTED_FILE_TYPES
    )

    st.markdown("---")

    st.markdown("### Features")

    st.success("✅ Multi-format Support")
    st.success("✅ Semantic Search")
    st.success("✅ AI Answers")
    st.success("✅ Fast Retrieval")

# ---------------------------
# FILE PROCESSING
# ---------------------------

if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:

    with st.spinner("Indexing document..."):

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        chunks, index = ingest_and_index(tmp_path)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []

    st.success(f"✅ Indexed: {uploaded_file.name}")

# ---------------------------
# CHAT WINDOW
# ---------------------------

if st.session_state.chunks is not None:

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-msg'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='bot-msg'>🤖 {msg['content']}</div>",
                unsafe_allow_html=True,
            )

    query = st.chat_input("Ask anything about your document...")

    if query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.spinner("Thinking..."):
            answer = answer_query(
                query,
                st.session_state.chunks,
                st.session_state.index,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.rerun()

else:

    st.markdown("""
<div class="upload-card">

# 👋 Welcome

Upload a document from the sidebar to start chatting.

### Supported

- PDF
- DOCX
- TXT
- PPTX

Once indexed, you can ask unlimited questions.

</div>
""", unsafe_allow_html=True)
