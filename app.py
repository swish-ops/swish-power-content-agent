"""Streamlit MVP for the Swish Power Content Agent."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import streamlit as st
from agents import Agent, Runner, WebSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")


def get_openai_api_key() -> str | None:
    """Return the configured OpenAI API key without exposing it in the UI."""

    api_key = os.getenv("OPENAI_API_KEY")
    return api_key.strip() if api_key and api_key.strip() else None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

BRAND_CONTEXT = """
Swish Power Solutions sells portable power stations, solar panels, battery gear,
and off-grid/camping power products in Australia.

Audience: Australian campers, caravanners, tradies, 4WD users, backup power
buyers, off-grid users, and people interested in solar/battery tech explained
simply.

Tone: Plain English, practical, Australian, educational, confident, not
corporate, and not overhyped.
"""

CAROUSEL_STRUCTURE = """
Create exactly 8 slides:
1. Hook
2. What happened
3. Why it matters
4. Bigger context
5. Real-world meaning
6. Caveat / what to watch
7. Simple takeaway
8. CTA

Each slide must include a headline, body copy, and visual direction.

Slide 8 CTA guidance:
- Prefer a soft CTA focused on saving, following, and value-first education.
- Avoid overly salesy wording unless the post is specifically a product or offer post.
- Preferred CTA examples: "Save this before your next trip", "Follow Swish Power Solutions for plain-English camping power guides", "Follow for more", and "Save this guide".
- Preferred Slide 8 format:
  Headline: Not sure what size you need?
  Body: Save this before your next trip. Follow Swish Power Solutions for plain-English camping power guides.
  CTA button: Follow for more
"""


class Slide(BaseModel):
    """A single carousel slide."""

    slide_number: int = Field(..., ge=1, le=8)
    role: str
    headline: str
    body_copy: str
    visual_direction: str


class ContentPack(BaseModel):
    """Final structured content package for Swish Power Solutions."""

    topic_title: str
    short_topic_summary: str
    source_links: list[str] = Field(default_factory=list)
    fact_check_notes: list[str] = Field(default_factory=list)
    carousel_slides: list[Slide] = Field(..., min_length=8, max_length=8)
    caption: str
    hashtags: list[str] = Field(..., min_length=3, max_length=5)
    visual_brief: str
    confidence_score: float = Field(..., ge=0, le=1)
    status: Literal["Draft"] = "Draft"


class ContentRequest(BaseModel):
    """User input passed to the agent workflow."""

    topic_input: str
    mode: Literal["Manual topic", "Find current topic ideas"]


def build_specialist_agents(model: str) -> dict[str, Agent]:
    """Create the specialist agents used by the manager agent."""

    shared_guardrails = (
        f"{BRAND_CONTEXT}\n{CAROUSEL_STRUCTURE}\n"
        "Keep claims specific, useful, and easy for Australians to understand. "
        "Do not invent statistics, laws, prices, or news. If a claim needs a "
        "source and you do not have one, mark it as a caveat."
    )

    topic_scout = Agent(
        name="Topic Scout Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Find or refine one practical content topic for Swish Power Solutions. "
            "If the user asks for current ideas, use web search and return a topic "
            "that is timely for Australian camping, caravanning, 4WD, backup power, "
            "solar, batteries, or portable power. Include source URLs when used."
        ),
        tools=[WebSearchTool()],
    )

    research = Agent(
        name="Research Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Research the chosen topic. Prioritise Australian relevance and primary "
            "or reputable sources. Summarise only the facts needed for a simple social "
            "carousel and include source URLs."
        ),
        tools=[WebSearchTool()],
    )

    fact_checker = Agent(
        name="Fact Checker Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Check the research and draft claims for accuracy, uncertainty, and "
            "missing context. Flag anything that should be softened, removed, or "
            "presented as general guidance rather than a hard claim."
        ),
    )

    carousel_writer = Agent(
        name="Carousel Writer Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Write exactly 8 carousel slides using the required structure. Make each "
            "slide concise and useful. Avoid hype, fearmongering, and salesy language."
        ),
    )

    brand_voice = Agent(
        name="Brand Voice Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Edit copy into Swish Power Solutions' voice: plain English, practical, "
            "Australian, educational, confident, not corporate, and not overhyped."
        ),
    )

    caption = Agent(
        name="Caption Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Write one Instagram/Facebook caption and 3-5 relevant hashtags. Keep the "
            "caption helpful, conversational, and Australian. Do not create image prompts."
        ),
    )

    slide_brief = Agent(
        name="Slide Brief Agent",
        model=model,
        instructions=(
            f"{shared_guardrails}\n"
            "Create clear visual directions for each slide plus an overall visual brief. "
            "Do not generate images; only describe practical visual/layout direction."
        ),
    )

    return {
        "topic_scout": topic_scout,
        "research": research,
        "fact_checker": fact_checker,
        "carousel_writer": carousel_writer,
        "brand_voice": brand_voice,
        "caption": caption,
        "slide_brief": slide_brief,
    }


def build_manager_agent(model: str) -> Agent:
    """Create the manager agent that coordinates specialist agents as tools."""

    specialists = build_specialist_agents(model)

    return Agent(
        name="Manager Agent",
        model=model,
        instructions=(
            f"{BRAND_CONTEXT}\n{CAROUSEL_STRUCTURE}\n"
            "You are the manager for the Swish Power Content Agent. Coordinate the "
            "specialist agents provided as tools and produce one final structured output.\n\n"
            "Workflow:\n"
            "1. Use the Topic Scout Agent to choose or refine the topic. If mode is "
            "'Find current topic ideas', ask it to search for current ideas.\n"
            "2. Use the Research Agent when web research, current context, or source "
            "links are needed. For a manual evergreen topic, research only if factual "
            "claims need support.\n"
            "3. Use the Fact Checker Agent to review claims and capture notes.\n"
            "4. Use the Carousel Writer Agent to draft the 8-slide carousel.\n"
            "5. Use the Brand Voice Agent to adjust language to the Swish Power voice.\n"
            "6. Use the Caption Agent for caption and hashtags.\n"
            "7. Use the Slide Brief Agent for slide visual directions and overall brief.\n\n"
            "Return the final ContentPack only. Include source_links if research was "
            "used. If no external research was used, source_links can be empty and "
            "fact_check_notes should explain that the pack uses general educational guidance. "
            "Always set status to Draft and use a confidence_score from 0 to 1."
        ),
        tools=[
            specialists["topic_scout"].as_tool(
                tool_name="topic_scout",
                tool_description="Finds or refines current and evergreen content topics.",
            ),
            specialists["research"].as_tool(
                tool_name="research_agent",
                tool_description="Researches Australian-relevant facts and source links.",
            ),
            specialists["fact_checker"].as_tool(
                tool_name="fact_checker",
                tool_description="Checks claims, caveats, and confidence.",
            ),
            specialists["carousel_writer"].as_tool(
                tool_name="carousel_writer",
                tool_description="Writes the 8-slide carousel copy.",
            ),
            specialists["brand_voice"].as_tool(
                tool_name="brand_voice",
                tool_description="Rewrites copy in Swish Power's plain-English Australian voice.",
            ),
            specialists["caption"].as_tool(
                tool_name="caption_agent",
                tool_description="Writes the social caption and hashtags.",
            ),
            specialists["slide_brief"].as_tool(
                tool_name="slide_brief",
                tool_description="Creates visual directions and the overall visual brief.",
            ),
        ],
        output_type=ContentPack,
    )


async def run_content_workflow(request: ContentRequest, model: str) -> ContentPack:
    """Run the manager agent and return the final content pack."""

    manager = build_manager_agent(model)
    prompt = (
        "Generate a Swish Power Solutions carousel content package for this request:\n"
        f"{request.model_dump_json(indent=2)}"
    )
    result = await Runner.run(manager, prompt)
    return result.final_output


def content_pack_to_markdown(pack: ContentPack) -> str:
    """Format a content pack as Markdown for export."""

    lines = [
        f"# {pack.topic_title}",
        "",
        f"**Status:** {pack.status}",
        f"**Confidence score:** {pack.confidence_score:.2f}",
        "",
        "## Short topic summary",
        pack.short_topic_summary,
        "",
        "## Source links",
    ]

    if pack.source_links:
        lines.extend(f"- {link}" for link in pack.source_links)
    else:
        lines.append("- No external source links used.")

    lines.extend(["", "## Fact-check notes"])
    lines.extend(f"- {note}" for note in pack.fact_check_notes)

    lines.extend(["", "## Carousel slides"])
    for slide in pack.carousel_slides:
        lines.extend(
            [
                "",
                f"### Slide {slide.slide_number}: {slide.role}",
                f"**Headline:** {slide.headline}",
                "",
                f"**Body copy:** {slide.body_copy}",
                "",
                f"**Visual direction:** {slide.visual_direction}",
            ]
        )

    lines.extend(
        [
            "",
            "## Caption",
            pack.caption,
            "",
            "## Hashtags",
            " ".join(pack.hashtags),
            "",
            "## Visual brief",
            pack.visual_brief,
            "",
        ]
    )
    return "\n".join(lines)


def safe_filename(title: str) -> str:
    """Create a safe Markdown filename from a topic title."""

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "content-pack"
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date}-{slug[:70]}.md"


def render_content_pack(pack: ContentPack) -> None:
    """Render the content pack in Streamlit."""

    st.subheader(pack.topic_title)
    st.write(pack.short_topic_summary)

    metrics = st.columns(2)
    metrics[0].metric("Status", pack.status)
    metrics[1].metric("Confidence", f"{pack.confidence_score:.0%}")

    with st.expander("Source links", expanded=bool(pack.source_links)):
        if pack.source_links:
            for link in pack.source_links:
                st.markdown(f"- {link}")
        else:
            st.write("No external source links used.")

    with st.expander("Fact-check notes", expanded=True):
        for note in pack.fact_check_notes:
            st.markdown(f"- {note}")

    st.subheader("8-slide carousel")
    for slide in pack.carousel_slides:
        with st.container(border=True):
            st.markdown(f"**Slide {slide.slide_number}: {slide.role}**")
            st.markdown(f"### {slide.headline}")
            st.write(slide.body_copy)
            st.caption(f"Visual direction: {slide.visual_direction}")

    st.subheader("Caption")
    st.write(pack.caption)
    st.write(" ".join(pack.hashtags))

    st.subheader("Visual brief")
    st.write(pack.visual_brief)

    st.subheader("Structured JSON")
    st.json(json.loads(pack.model_dump_json()))

    markdown = content_pack_to_markdown(pack)
    st.download_button(
        "Export Markdown",
        data=markdown,
        file_name=safe_filename(pack.topic_title),
        mime="text/markdown",
    )


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="Swish Power Content Agent", page_icon="⚡")
    st.title("⚡ Swish Power Content Agent")
    st.caption("MVP carousel content-pack generator for Instagram/Facebook.")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
        api_key = get_openai_api_key()
        if not api_key:
            st.info("Add your OPENAI_API_KEY to .env before generating content.")

    mode = st.radio(
        "Input type",
        options=["Manual topic", "Find current topic ideas"],
        horizontal=True,
    )

    default_prompt = (
        "Portable power safety tips for a long weekend camping trip"
        if mode == "Manual topic"
        else "Find a timely Australian camping, solar, battery, or backup power topic."
    )
    topic_input = st.text_area("Topic input", value=default_prompt, height=120)

    if st.button("Generate Content Pack", type="primary"):
        if not topic_input.strip():
            st.error("Please enter a topic or current-topic request.")
            return
        if not get_openai_api_key():
            st.error("Add your OPENAI_API_KEY to .env before generating content.")
            return

        request = ContentRequest(topic_input=topic_input.strip(), mode=mode)
        with st.spinner("Manager Agent is coordinating the specialist agents..."):
            try:
                pack = asyncio.run(run_content_workflow(request, model.strip() or DEFAULT_MODEL))
            except Exception as exc:  # Streamlit UI boundary: show actionable errors to the user.
                st.error("Content generation failed. Check your API key, model access, and network connection.")
                st.exception(exc)
                return

        st.session_state["content_pack"] = pack

    if "content_pack" in st.session_state:
        render_content_pack(st.session_state["content_pack"])


if __name__ == "__main__":
    main()
