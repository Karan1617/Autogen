'''
backend.py
=================

Main code for a two-agent literature review assistant using AutoGen (v0.4+).

It has one main function: `run_litrev()`

The two agents:
1. search_agent - searches for papers using arXiv.
2. summarizer - writes a short summary from the found papers.

This file works on its own and can be used in apps like CLI, Streamlit, FastAPI, Gradio, etc.

'''

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, List

import arxiv 
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Tool definition 
def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Returns a short list of arXiv papers that match the given *query*.

Each paper includes: title, authors, published date, summary, and PDF link.

This function is set up as an AutoGen tool, so agents can use it directly.

    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers

arxiv_tool = FunctionTool(
    arxiv_search,
    description=(
       """
            Searches arXiv and returns up to *max_results* papers.

            Each paper includes: title, authors, published date, abstract, and PDF link.
       """

    ),
)

#  Agents 
def build_team(model: str = "gpt-4o-mini") -> RoundRobinGroupChat:
    """Create and return a two-agent *RoundRobinGroupChat* team."""
    llm_client = OpenAIChatCompletionClient(model=model,api_key='your-key-here')
    # Agent that **only** calls the arXiv tool and forwards top‑N papers
    search_agent = AssistantAgent(
        name="search_agent",
        description="Crafts arXiv queries and retrieves candidate papers.",
        system_message=(
            "Given a user topic, think of the best arXiv query and call the "
            "provided tool. Always fetch five-times the papers requested so "
            "that you can down-select the most relevant ones. When the tool "
            "returns, choose exactly the number of papers requested and pass "
            "them as concise JSON to the summarizer."
        ),
        tools=[arxiv_tool],
        model_client=llm_client,
        reflect_on_tool_use=True,
    )

    # Agent that writes the final literature review
    summarizer_agent = AssistantAgent(
        name="summarizer_agent",
        description="Produces a short Markdown review from provided papers.",
        system_message=(
            """
                You are a research expert. Given a JSON list of papers, write a short literature review in Markdown:

                1. Start with a 2-3 sentence intro about the topic.
                2. For each paper, add a bullet with:
                     - Title (as a Markdown link)
                     - Authors
                     - The problem it addresses
                     - Its main contribution
                3. End with one sentence that sums up the key takeaway.

            """

        ),
        model_client=llm_client,
    )

    return RoundRobinGroupChat(
        participants=[search_agent, summarizer_agent],
        max_turns=2,
    )


async def run_litrev(
    topic: str,
    num_papers: int = 5,
    model: str = "gpt-4o-mini",
) -> AsyncGenerator[str, None]:
    """Yield strings representing the conversation in real-time."""

    team = build_team(model=model)
    task_prompt = (
        f"Conduct a literature review on **{topic}** and return exactly {num_papers} papers."
    )

    async for msg in team.run_stream(task=task_prompt):
        if isinstance(msg, TextMessage):
            yield f"{msg.source}: {msg.content}"

# testing /demo
if __name__ == "__main__":
    async def _demo() -> None:
        async for line in run_litrev("neural networks for biochemistry", num_papers=5):
            print(line)

    asyncio.run(_demo())
