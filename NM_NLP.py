import streamlit as st
from langchain_core.output_parsers import StrOutputParser #Convert the JSON form Output into Normal Output form
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
import warnings

warnings.filterwarnings("ignore")
# 💬 Load LLM (you can swap for Gemini if needed)

api_key=st.sidebar.text_input("Load your Groq Api_key")


select=st.sidebar.selectbox("select the model", ["gemma2-9b-it","llama-3.3-70b-versatile"])
if api_key:
    LLM = ChatGroq(
            api_key=api_key,
            model=select,
            temperature=1
        )
else:
    st.error("Load your API key")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="""your job is to reply in a *different language* that the user can understand (as specified).
"""),
    ]
 

# 📄 Prompt template



prompt = ChatPromptTemplate.from_template(
    """
You are a friendly, multilingual chatbot. The user will speak in any language, and your job is to reply in a *different language* that the user can understand (as specified).

Do *not* translate word-by-word — instead, have a natural conversation in the *target language* that matches the meaning and mood of the message.

User's message (in {input_language}):
{message}

Respond conversationally in only on  {output_language} do not use anohter language.
"""
)



# 🧠 Build chain
if api_key:
    chain = prompt | LLM | StrOutputParser()

# 🚀 Streamlit UI

st.title("🌍 Multilingual Chatbot with LangChain")
 
# 📌 Language selection
languages = ["ENGILISH", "TAMIL", "HINDHI", "FRENCH", "GERMAN", "JAPANESE"]
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("Input Language", languages, index=0)
with col2: 
    output_lang = st.selectbox("Output Language", languages, index=1)

# 📝 User input
user_input = st.chat_input("Enter your message") 
if user_input:

    human_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(human_message)


if user_input:
    with st.spinner("Translating..."):
        result = chain.invoke({
                "input_language": input_lang,
                "output_language": output_lang,
                "message": user_input
        })
        st.session_state.chat_history.append(AIMessage(content=result))
        st.session_state.chat_history = st.session_state.chat_history[-100:]
        

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"AI: {message.content}")

st.button("Clear Chat", on_click=lambda: st.session_state.chat_history.clear())