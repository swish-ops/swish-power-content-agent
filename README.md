# Swish Power Content Agent

A simple MVP web app that uses the OpenAI Agents SDK to generate Instagram/Facebook carousel content packages for Swish Power Solutions.

## What it does

The app lets you enter either:

1. A manual content topic, or
2. A request to find current topic ideas.

It produces a structured draft content package with:

- Topic title
- Short topic summary
- Source links when research is used
- Fact-check notes
- 8-slide carousel copy
- Caption
- 3-5 hashtags
- Visual brief
- Confidence score
- Status set to `Draft`

## Agent workflow

The Manager Agent coordinates these specialist agents through the OpenAI Agents SDK:

- Topic Scout Agent
- Research Agent
- Fact Checker Agent
- Carousel Writer Agent
- Brand Voice Agent
- Caption Agent
- Slide Brief Agent

The Topic Scout and Research agents can use web search for timely/current topics. V1 only generates the content package; it does not post, schedule, generate images, manage users, take payments, or provide analytics.

## Carousel structure

Each generated carousel contains 8 slides:

1. Hook
2. What happened
3. Why it matters
4. Bigger context
5. Real-world meaning
6. Caveat / what to watch
7. Simple takeaway
8. CTA

Each slide includes:

- Headline
- Body copy
- Visual direction

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the local app

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

## Using the app

1. Choose `Manual topic` or `Find current topic ideas`.
2. Enter your topic or current-topic request.
3. Click `Generate Content Pack`.
4. Review the structured output on the page.
5. Click `Export Markdown` to download the content pack as a Markdown file.

## Notes for editing later

- The Streamlit UI, agent definitions, Pydantic output schema, and Markdown export live in `app.py`.
- Change the default model in `app.py` or type a model name in the Streamlit sidebar.
- Keep the specialist agent prompts focused and short while the product direction is still evolving.
