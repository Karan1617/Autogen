'''
frontend.py
================

A simple Streamlit interface for the literature review assistant from `backend.py`.

Users enter a topic and number of papers, then see the two agents work together in real-time.

'''

import asyncio
import numpy as np
import streamlit as st

from backend import run_litrev

st.set_page_config(page_title="Literature Review Assistant")
st.title("Literature Review Assistant")

query = st.text_input("Research topic")
n_papers = st.slider("Number of papers", 1, 10, 5)

if st.button("Search") and query:

    async def _runner() -> None:
        chat_placeholder = st.container()
        async for frame in run_litrev(query, num_papers=n_papers):
            role, *rest = frame.split(":", 1)
            content = rest[0].strip() if rest else ""
            with chat_placeholder:
                with st.chat_message("assistant"):
                    st.markdown(f"**{role}**: {content}")

    with st.spinner("Working..."):
        try:
            asyncio.run(_runner())
        except RuntimeError:
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_runner())

    st.success("Review completed!")
