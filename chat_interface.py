import streamlit as st
from api_utils import get_api_response

def display_chat_interface():
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get API response
        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                with st.chat_message("assistant"):
                    st.markdown(response['answer'])

                with st.expander("Details"):
                    st.subheader("Generated Answer")
                    st.code(response['answer'])
                    st.subheader("Model Used")
                    st.code(response['model'])
                    st.subheader("Session ID")
                    st.code(response['session_id'])

                    # Correct place to show sources
                    if "retrieved_chunks" in response:
                        st.subheader("Sources")
                        for idx, chunk in enumerate(response["retrieved_chunks"]):
                            st.markdown(
                                f"**Source:** `{chunk.get('source', 'unknown')}` | **Chunk ID:** `{chunk.get('chunk_id', 'n/a')}`"
                            )
                            st.markdown(f"```text\n{chunk.get('content', '')}\n```")

            else:
                st.error("Failed to get a response from the API. Please try again.")
