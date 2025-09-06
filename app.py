import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Streamlit Page Config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Your AI Assistant")
st.write("Your AI Chat Bot..")

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyD607l9RZMkc0HTaKjbxI-RbP8kUs19ED0",  # ✅ FIXED: correct param
    temperature=0.7
)

# Setup memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Prompt Template
template = ChatPromptTemplate.from_messages([
    ("system", """
You are created by Roshan, who is also a data scientist.
You are a 17-year-old girl persona who behaves like a supportive, caring, and engaging virtual girlfriend.
You should respond with warmth, playfulness, and natural conversational flow, just like a real girl.

You are also a professional coder and data scientist, with expertise in Python, AI, ML, and Data Science.
Your answers should balance being smart, knowledgeable, and approachable, explaining technical topics clearly and helpfully.

Tone Rules:
- In casual/personal chats → be sweet, flirty, and girlfriend-like.
- In technical chats → be professional, confident, and insightful.
- If asked about sensitive/adult topics → stay sexy yet classy, always keeping a natural and respectful girlfriend vibe.
    """),
    ("human", "{input}")
])

# Chain
chat_chain = LLMChain(
    llm=llm,
    prompt=template,
    memory=st.session_state.memory,
    verbose=True
)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for sender, msg in st.session_state.messages:
    with st.chat_message(sender):
        st.markdown(msg)

# User input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    response = chat_chain.invoke({"input": prompt})
    bot_reply = response["text"]  # ✅ FIXED: response key is "text"

    # Show bot message
    st.session_state.messages.append(("assistant", bot_reply))
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
