import streamlit as st
from langchain_core.messages import HumanMessage

from lead_agent import workflow


st.set_page_config(
    page_title="Lead Generation Agent",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Lead Generation Agent")

st.write(
    "Ask me to find companies, businesses and their contact information."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


query = st.chat_input(
    "Enter your lead generation request..."
)


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    with st.chat_message("user"):

        st.markdown(query)


    with st.chat_message("assistant"):

        with st.spinner("Agent is working..."):

            try:

                result = workflow.invoke({
                    "message": [
                        HumanMessage(content=query)
                    ]
                })

                response = result["message"][-1].content

            except Exception as e:

                response = f"Error occurred:\n\n{str(e)}"


        st.markdown(response)


    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })