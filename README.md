# Real-Time Research Agent

An end-to-end research assistant that pulls live information from the public web, recent news, and LLM reasoning to produce a concise, source-backed briefing. The app is built with Streamlit, LangChain, LangGraph, Google Gemini 2.5 Flash, and the `ddgs` DuckDuckGo client.

---

## Key Features

- **Multi-source evidence** -- combines DuckDuckGo web search (`ddgs`), DuckDuckGo News, and News API articles.
- **LLM synthesis** -- uses Gemini 2.5 Flash (via LangChain) orchestrated through LangGraph's tool-calling workflow.
- **Structured reports** -- generates a markdown document with Executive Summary, Overview, Recent Developments, Key Facts, Perspectives, Takeaways, References, and Research Notes.
- **Agent reasoning stream** -- optionally shows real-time thoughts, tool calls, and observations while the run is in progress.
- **Fallback synthesis** -- gracefully retries with a direct prompt if the LangGraph agent fails or returns malformed output.
- **Export options** -- download the analysis as Markdown or PDF.

---

## Tech Stack

| Layer          | Technology                                  |
| -------------- | ------------------------------------------- |
| UI             | Streamlit                                   |
| Agent runtime  | LangGraph + LangChain                       |
| LLM            | Google Gemini 2.5 Flash                     |
| Search & news  | `ddgs` DuckDuckGo client, DuckDuckGo News, News API |
| PDF rendering  | `fpdf2`                                     |
| Environment    | Python 3.10+                                |

---

## Prerequisites

- Python 3.10 or higher  
- API keys:
  - `GOOGLE_API_KEY` (required - Gemini via AI Studio)
  - `NEWS_API_KEY` (optional but recommended - [NewsAPI.org](https://newsapi.org))

---

## Local Setup

```bash
git clone https://github.com/<your-username>/Real-Time-Research-Agent.git
cd Real-Time-Research-Agent
pip install -r requirements.txt
```

Create a `.env` file (kept out of Git via `.gitignore`):

```env
GOOGLE_API_KEY=your_gemini_key
NEWS_API_KEY=your_news_api_key  # optional
```

Run the app:

```bash
streamlit run agent.py
```

The UI launches at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).  
2. In Streamlit Cloud, create a new app referencing `agent.py`.  
3. Add your secrets under **App -> Settings -> Secrets**:

```
GOOGLE_API_KEY="your_value"
NEWS_API_KEY="your_value"
```

No `.env` file is read in the cloud environment; Streamlit injects these values as environment variables so the existing `load_dotenv()` + `os.getenv()` pattern still works.

---

## Usage

1. Enter any person, place, organization, or concept in the input box.
2. Click **Research**. The agent will:
   - kick off LangGraph workflow,
   - collect info via DuckDuckGo + News tools,
   - synthesize a markdown report with references.
3. Toggle **Show Agent Reasoning** to stream intermediate thoughts/tool calls.
4. Download the resulting report as Markdown or PDF.

Example prompts: *"Shah Rukh Khan"*, *"Next-gen lithium battery startups"*, *"Tokyo tourism 2025"*.

---

## Project Structure

```
Real-Time-Research-Agent/
|-- agent.py            # Streamlit app + LangGraph agent
|-- requirements.txt    # Python dependencies
|-- README.md           # Documentation (this file)
`-- .env                # Local secrets (not committed)
```

---

## Configuration Notes

- **LLM model**: change the model name in `agent.py` (`ChatGoogleGenerativeAI(model="gemini-2.5-flash")`) if you want a different Gemini variant.
- **Search depth**: update `max_results` arguments inside `search_web`, `get_news`, and `duckduckgo_news`.
- **Report template**: edit the `RESEARCH_AGENT_PROMPT` constant to tweak section names or instructions.
- **Reasoning stream**: toggled via the "Show Agent Reasoning" checkbox in the sidebar.

---

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |
| `GOOGLE_API_KEY` missing error | Add the key to `.env` (local) or Streamlit Secrets (cloud). |
| News API quota reached | The app falls back to DuckDuckGo News, but expect fewer recent-article insights. |
| Streamlit not opening | `streamlit run agent.py` must be executed inside an environment where the dependencies were installed. |

---

## License

Released under the MIT License. See `LICENSE` (add one if missing) for details.

---

## Acknowledgments

- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) for the agent tooling.
- [Google Gemini](https://deepmind.google/technologies/gemini/) for high-quality, fast LLM responses.
- [DuckDuckGo Search (`ddgs`)](https://pypi.org/project/ddgs/) and [News API](https://newsapi.org) for real-time data.
- [Streamlit](https://streamlit.io/) for the rapid UI.

Built with love to make comprehensive research effortless.
