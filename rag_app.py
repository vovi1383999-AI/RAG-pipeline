import streamlit as st

# 1. Khởi tạo session state để lưu lịch sử chat nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Hiển thị lịch sử cũ (không tốn quota)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chỉ gọi API khi người dùng thực sự nhập liệu mới
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị câu hỏi người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Gọi API (Lúc này mới tốn 1 quota)
    try:
        response = model.generate_content(prompt)
        bot_reply = response.text
        
        # Hiển thị và lưu câu trả lời
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        
    except Exception as e:
        st.error(f"Hết lượt dùng: {e}")


import google.generativeai as genai
from pinecone import Pinecone

# 1. Cấu hình trang
st.set_page_config(page_title="Chatbot Nhân Sự (RAG Demo)", layout="centered")
st.title("🤖 Trợ lý HR thông minh")
st.caption("Hỏi đáp dựa trên quy định nội bộ (Dữ liệu từ Pinecone)")

# 2. Sidebar: Nhập Key (Để kết nối 2 đầu mối)
with st.sidebar:
    st.header("🔐 Cấu hình API")
    google_api_key = st.text_input("Google API Key:", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key:", type="password")
    index_name = st.text_input("Tên Index Pinecone:", value="demo-rag-it1994")
    
    st.info("Lưu ý: Phải dùng đúng Key và Tên Index bạn đã tạo trên Colab.")

# 3. Hàm xử lý logic (Core Functions)
def get_embedding(text):
    """Biến câu hỏi thành Vector (giống hệt lúc nạp dữ liệu)"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query" # Lưu ý: type là query
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Lỗi tạo vector: {e}")
        return None

def query_pinecone(vector, index_name, api_key):
    """Gửi vector lên Pinecone để tìm kiếm"""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    results = index.query(
        vector=vector,
        top_k=3, # Lấy 3 đoạn văn bản liên quan nhất
        include_metadata=True
    )
    return results

# 4. Giao diện Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý khi người dùng nhập câu hỏi
if prompt := st.chat_input("Bạn muốn hỏi gì về quy định công ty?"):
    # Hiện câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kiểm tra Key
    if not google_api_key or not pinecone_api_key:
        st.error("Vui lòng nhập đủ API Key bên tay trái!")
        st.stop()

    # AI Xử lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # BƯỚC 1: Cấu hình Google
            genai.configure(api_key=google_api_key)
            
            # BƯỚC 2: Tìm kiếm ngữ cảnh (Retrieval)
            message_placeholder.markdown("🔍 *Đang tra cứu tài liệu...*")
            question_vector = get_embedding(prompt)
            
            if question_vector:
                search_results = query_pinecone(question_vector, index_name, pinecone_api_key)
                
                # Tổng hợp thông tin tìm được thành 1 đoạn văn (Context)
                context_text = ""
                for match in search_results['matches']:
                    # Chỉ lấy tin có độ tin cậy > 50%
                    if match['score'] > 0.5: 
                        context_text += f"- {match['metadata']['text_content']}\n"
                
                if not context_text:
                    context_text = "Không tìm thấy thông tin cụ thể trong tài liệu."

                # BƯỚC 3: Tạo Prompt cho AI (Augmented Generation)
                # Đây là kỹ thuật "Grounding" - Ép AI chỉ trả lời dựa trên Context
                final_prompt = f"""
                Bạn là trợ lý nhân sự. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
                Nếu thông tin không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trong quy định".
                Đừng tự bịa ra thông tin.

                Thông tin ngữ cảnh (Context):
                {context_text}

                Câu hỏi của người dùng:
                {prompt}
                """

                # BƯỚC 4: Gửi cho Gemini trả lời
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(final_prompt)
                
                full_response = response.text
                message_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = f"Có lỗi xảy ra: {str(e)}"
            message_placeholder.error(full_response)
            
    # Lưu lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})
