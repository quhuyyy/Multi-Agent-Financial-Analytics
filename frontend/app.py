import streamlit as st
import requests
from datetime import datetime

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="IUH - Multi-Agent System for Vietnamese Equity Analysis", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main layout adjustments */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        max-width: 100%;
    }
    
    .stApp {
        background-color: #f0f2f6;
        overflow: hidden;
    }
    
    .main {
    }

    
    /* Header styling */
    .header-section {
        background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        text-align: center;
        color: white;
    }
    
    .header-section h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .header-section p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Chat messages container - Row 1 */
    .chat-messages {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        height: calc(100vh - 400px);
        overflow-y: auto;
        flex: 1;
    }
    
    /* Chat input container - Row 2 */
    .chat-input-container {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Chat container styling */
    .stContainer {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        height: calc(100vh - 400px);
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }
    
    /* Custom scrollbar for chat container */
    .stContainer::-webkit-scrollbar {
        width: 8px;
    }
    
    .stContainer::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    .stContainer::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    
    .stContainer::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    /* Chat input section - inside chat container */
    .chat-input-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        border: 2px solid #4f46e5;
        margin-top: 1rem;
        position: relative;
        z-index: 10;
    }
    
    .chat-input-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
    }
    
    /* Author cards */
    .author-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
        border: 1px solid #93c5fd;
        transition: transform 0.2s;
    }
    
    .author-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.3);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #93c5fd;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Rộng hơn khi sidebar đang mở */
    [data-testid="stSidebar"][aria-expanded="true"] {
    width: 370px !important;       /* đổi số tuỳ bạn: 420/440/480px */
    min-width: 370px !important;
    max-width: 370px !important;
    }

    /* Khi sidebar đóng thì width = 0 như mặc định */
    [data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
    min-width: 0px !important;
    }

    /* Tuỳ chọn: chỉ áp dụng trên màn lớn, nhỏ hơn thì giữ mặc định */
    @media (max-width: 1024px) {
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    }
    
    /* Custom scrollbar for chat messages */
    .chat-messages::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    /* Hide main page scrollbar */
    body {
        overflow: hidden;
    }
    
    /* Ensure no scrolling on main elements */
    .stApp > div {
        overflow: hidden;
    }
    
    /* Special styling for chat input */
    .stChatInput > div > div > input {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #4f46e5;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #06b6d4;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        outline: none;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #64748b;
        font-style: italic;
    }
    
    /* Special button styling */
    .stChatInput > div > div > button {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div > div > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
    }
    
    /* Make col1 a flex container for 2 rows */
    .main .block-container > div:first-child {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 200px);
    }
    
    .main .block-container > div:first-child > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    
    /* Adjust main content to account for fixed input */
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Section frames */
    .section-frame {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #93c5fd;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.1);
        display: block;
    }
    
    .section-frame h3 {
        margin: 0 0 1rem 0;
        color: #1e40af;
        border-bottom: 2px solid #93c5fd;
        padding-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .section-frame h4 {
        margin: 0 0 0.5rem 0;
        color: #1e40af;
        border-bottom: 1px solid #93c5fd;
        padding-bottom: 0.3rem;
        font-size: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Fixed header
st.markdown("""
<div class="header-section">
    <h1>🤖 IUH - Multi-Agent System for Vietnamese Equity Analysis</h1>
    <p>Multi-Agent AI System for Vietnamese Stock Analysis - Industrial University of Ho Chi Minh City</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>👥 Authors</h3>
        <div class="author-card">
            <strong>Le Quoc Huy</strong><br>
            <small>📧 22636191.huy@iuh.edu.vn</small>
        </div>
        <div class="author-card">
            <strong>Nguyen Thanh Trong</strong><br>
            <small>📧 22642481.trong@student.iuh.edu.vn</small>
        </div>
        <div class="author-card">
            <strong>Pham Gia Khanh</strong><br>
            <small>📧 22724051.khanh@student.iuh.edu.vn</small>
        </div>
        <div class="author-card">
            <strong>Nguyen Minh Phuc</strong><br>
            <small>📧 22637001.phuc@student.iuh.edu.vn</small>
        </div>
        <div class="author-card">
            <strong>Khuong N Dang</strong><br>
            <small>📧 23741531.khuong@student.iuh.edu.vn</small><br>
            <small>🔗 ORCID: 0009-0006-1584-2499</small>
        </div>
        <div class="author-card">
            <strong>Pham Thi Thiet</strong><br>
            <small>📧 phamthithiet@iuh.edu.vn</small>
        </div>
        <div class="author-card">
            <strong>Nguyen Chi Kien</strong><br>
            <small>📧 nguyenchikien@iuh.edu.vn</small><br>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content area

content = st.container()

with content:
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and isinstance(message["content"], dict):
                    # In text nếu có
                    if message["content"].get("answer"):
                        st.markdown(message["content"]["answer"])
                    # In ảnh nếu có
                    if message["content"].get("images"):
                        for img_url in message["content"]["images"]:
                                st.image(img_url, caption="📊 Analysis chart from system")
                else:
                    st.markdown(message["content"])
    
    user_question = st.chat_input("Ask any question about Vietnamese stock analysis...")
    
    
    # Handle user input
    if user_question:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message in chat container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)

        # Show loading spinner
            with st.spinner("🤖 AI is analyzing your question..."):
                backend_url = "http://backend:8000/ask"
                payload = {"question": user_question, "history": st.session_state.messages}

                try:
                    response = requests.post(backend_url, json=payload, timeout=200)
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        images = data.get("images", [])

                        with st.chat_message("assistant"):
                            if answer:
                                st.markdown(answer)
                            if images:
                                for img_url in images:
                                    st.image(img_url, caption="📊 Analysis chart from system")

                        # Save both text and images for proper history display
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": {"answer": answer, "images": images}
                        })

                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("⏰ System response is slow. Please try again later.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to backend. Please check your network connection.")
                except Exception as e:
                    st.error(f"⚠️ Unknown error: {str(e)}")
    
# Close main container
st.markdown('</div>', unsafe_allow_html=True)
