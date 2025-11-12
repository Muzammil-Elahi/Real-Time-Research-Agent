"""
Real-Time Research Agent
========================
This is an AI-powered research agent that uses LangChain/LangGraph to orchestrate
an agent that can search the web, gather news, and synthesize information into
structured research reports.

Architecture:
- Frontend: Streamlit (web UI)
- Agent Framework: LangGraph (orchestrates agent workflow)
- LLM: Google Gemini 2.5 Flash (via LangChain)
- Tools: DuckDuckGo Search, News API, DuckDuckGo News
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated, Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from operator import add
from duckduckgo_search import DDGS
from newsapi import NewsApiClient
import os
import copy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fpdf import FPDF
import markdown
import html

# Mapping of unsupported smart punctuation to ASCII equivalents
SMART_CHAR_MAP = str.maketrans({
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    "•": "-",
})

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================
# Load environment variables from .env file
# This allows us to store API keys securely without hardcoding them
load_dotenv()

# Get API keys from environment variables
# These are required for the LLM and News API services
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize the LLM (Large Language Model)
# Using Google Gemini 2.5 Flash - a fast, efficient model for this use case
# The model is bound with tools later to enable function calling
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Track tool references for transparency and UI display
CURRENT_TOOL_REFERENCES: List[Dict[str, Any]] = []
MAX_REFERENCES_PER_TOOL = 5


def reset_tool_references():
    """Clear tracked tool references before each new research run."""
    global CURRENT_TOOL_REFERENCES
    CURRENT_TOOL_REFERENCES = []


def log_tool_reference(tool_name: str, query: str, entries: List[Dict[str, str]]):
    """Record the sources returned by a tool for later display."""
    global CURRENT_TOOL_REFERENCES
    if not entries:
        return
    CURRENT_TOOL_REFERENCES.append(
        {
            "tool": tool_name,
            "query": query,
            "entries": entries[:MAX_REFERENCES_PER_TOOL],
        }
    )


def format_tool_references_for_prompt() -> str:
    """Create a plain-text summary of tool references for LLM instructions."""
    if not CURRENT_TOOL_REFERENCES:
        return "No references logged."
    lines = []
    for ref in CURRENT_TOOL_REFERENCES:
        entry_text = "; ".join(
            f"{item.get('title', 'Untitled Source')} - {item.get('url', 'No URL')}"
            for item in ref["entries"]
        )
        lines.append(f"{ref['tool']} (query: {ref['query']}): {entry_text}")
    return "\n".join(lines)

# ============================================================================
# TOOL FUNCTIONS
# ============================================================================
# These functions are wrapped as LangChain Tools that the agent can call
# Each function performs a specific task (web search, news search, etc.)
# The agent decides when and how to use these tools based on the query

def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo for comprehensive information.
    
    This is the primary tool for gathering general background information.
    DuckDuckGo is used because it's privacy-focused and doesn't require API keys.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and snippets
    """
    try:
        # Initialize DuckDuckGo search client
        ddgs = DDGS()
        
        # Perform text search (general web search, not news-specific)
        # max_results=10 limits the number of results for efficiency
        results = ddgs.text(query, max_results=10)
        
        if not results:
            return "No web results found."
        
        # Format results into a readable string
        # This format helps the LLM understand and process the information
        search_summary = "Web Search Results:\n\n"
        entries = []
        for idx, result in enumerate(results, 1):
            search_summary += f"{idx}. Title: {result['title']}\n"
            search_summary += f"   URL: {result['href']}\n"
            search_summary += f"   Snippet: {result['body']}\n\n"
            entries.append(
                {
                    "title": result.get("title", "Untitled"),
                    "url": result.get("href", ""),
                }
            )
        
        log_tool_reference("Web Search (DuckDuckGo)", query, entries)
        
        return search_summary
    except Exception as e:
        # Return error message if search fails
        # This prevents the agent from crashing and allows it to try other tools
        return f"Error searching web: {str(e)}"

def get_news(query: str) -> str:
    """
    Search for recent news articles using News API.
    
    News API provides access to articles from major publications worldwide.
    This tool is used to find recent developments and current events.
    
    Note: Requires NEWS_API_KEY to be set in environment variables.
    If not set, this function will fail gracefully.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing news articles with titles, sources, dates, and URLs
    """
    try:
        # Initialize News API client with API key
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Search for articles matching the query
        # get_everything() searches across all articles (not just headlines)
        # sort_by='publishedAt' ensures we get the most recent articles first
        articles = newsapi.get_everything(
            q=query,
            language='en',  # English only
            sort_by='publishedAt',  # Sort by publication date (newest first)
            page_size=10  # Limit to 10 articles
        )
        
        if articles['totalResults'] == 0:
            return "No recent news articles found."
        
        # Format results for the LLM to process
        news_summary = "Recent News Articles:\n\n"
        entries = []
        for idx, article in enumerate(articles['articles'][:10], 1):
            news_summary += f"{idx}. Title: {article['title']}\n"
            news_summary += f"   Source: {article['source']['name']}\n"
            news_summary += f"   Date: {article['publishedAt']}\n"
            # Use .get() with default to handle missing descriptions
            news_summary += f"   Description: {article.get('description', 'No description')}\n"
            news_summary += f"   URL: {article['url']}\n\n"
            entries.append(
                {
                    "title": article.get("title", "Untitled"),
                    "url": article.get("url", ""),
                }
            )
        
        log_tool_reference("News API", query, entries)
        
        return news_summary
    except Exception as e:
        error_text = str(e)
        lowered = error_text.lower()
        if "rate limit" in lowered or "429" in lowered:
            fallback_results = duckduckgo_news(query)
            return (
                "News API rate limit reached. Fallback to DuckDuckGo News:\n\n"
                f"{fallback_results}"
            )
        return f"Error fetching news: {error_text}"

def duckduckgo_news(query: str) -> str:
    """
    Search for recent news using DuckDuckGo News.
    
    This is an alternative news source to complement News API.
    DuckDuckGo News aggregates news from various sources and provides
    diverse perspectives. It doesn't require an API key.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing news articles from DuckDuckGo
    """
    try:
        # Initialize DuckDuckGo search client
        ddgs = DDGS()
        
        # Search specifically for news articles (not general web results)
        news_results = ddgs.news(query, max_results=10)
        
        if not news_results:
            return "No news results found."
        
        # Format results similar to other tools for consistency
        news_summary = "DuckDuckGo News Results:\n\n"
        entries = []
        for idx, article in enumerate(news_results, 1):
            news_summary += f"{idx}. Title: {article['title']}\n"
            news_summary += f"   Source: {article['source']}\n"
            news_summary += f"   Date: {article['date']}\n"
            news_summary += f"   Snippet: {article['body']}\n"
            news_summary += f"   URL: {article['url']}\n\n"
            entries.append(
                {
                    "title": article.get("title", "Untitled"),
                    "url": article.get("url", ""),
                }
            )
        
        log_tool_reference("DuckDuckGo News", query, entries)
        
        return news_summary
    except Exception as e:
        return f"Error fetching DDG news: {str(e)}"

def clean_markdown_output(raw_output: str) -> str:
    """
    Clean and format the agent's output to ensure proper markdown formatting.
    
    The LLM sometimes wraps its output in code blocks or includes prefixes.
    This function removes unwanted formatting while preserving the actual content.
    
    This is important because:
    1. The agent might wrap output in ```markdown``` code blocks
    2. The agent might include "Final Answer:" prefix
    3. There might be leading text before the first heading
    
    Args:
        raw_output: The raw output string from the LLM
        
    Returns:
        Cleaned markdown string ready for display
    """
    # Remove markdown code blocks if present
    # LLMs sometimes wrap markdown in code blocks, which we need to remove
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```markdown"):
        cleaned_output = cleaned_output[11:]  # Remove ```markdown
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]  # Remove ```
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]  # Remove trailing ```
    
    # Remove "Final Answer:" prefix if present
    # Some LLM responses include this prefix which we don't want in the final output
    if cleaned_output.startswith("Final Answer:"):
        cleaned_output = cleaned_output[13:].strip()
    elif cleaned_output.startswith("Final Answer"):
        cleaned_output = cleaned_output[12:].strip()
    
    # Remove any leading text before the first markdown heading
    # The report should start with "## Executive Summary"
    # If there's text before it, we remove it (but only if it's near the start)
    first_heading = cleaned_output.find("##")
    if first_heading > 0 and first_heading < 100:  # Only remove if heading is near the start
        cleaned_output = cleaned_output[first_heading:].strip()
    
    # Strip leading/trailing whitespace but preserve internal formatting
    return cleaned_output.strip()


def normalize_text_characters(text: str) -> str:
    """Replace smart quotes/dashes with ASCII equivalents for PDF compatibility."""
    if not isinstance(text, str):
        text = str(text)
    return text.translate(SMART_CHAR_MAP)


def extract_message_content(message: BaseMessage | str | None) -> str:
    """
    Safely convert a LangChain message (or raw string) into plain text.

    LangChain can return message content as strings, lists of content chunks,
    or other data structures. This helper normalizes the content so the UI
    logic and fallback synthesis can work with a clean string representation.
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()

    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def run_direct_research_synthesis(query: str) -> tuple[str, str]:
    """
    Fallback pipeline that gathers research data manually and asks the LLM
    to synthesize the final report without relying on LangGraph tool-calling.

    This is used when the tool-enabled agent fails to return a usable response
    (e.g., due to MALFORMED_FUNCTION_CALL or empty content).

    Returns:
        report_text: Cleaned markdown-ready report
        raw_output:  Raw LLM response before markdown cleanup
    """
    findings: Dict[str, str] = {
        "Web Search Results": search_web(query),
        "News API Results": get_news(query),
        "DuckDuckGo News Results": duckduckgo_news(query),
    }

    combined_findings = "\n\n".join(
        f"### {section}\n{details}"
        for section, details in findings.items()
    )

    synthesis_instruction = (
        f"You already gathered the following findings about '{query}'. "
        "Use ONLY this information to create the final markdown report exactly as defined in the OUTPUT FORMAT. "
        "If a section lacks evidence, write 'No verified information available.' instead of inventing facts."
    )

    reference_text = format_tool_references_for_prompt()

    response = llm.invoke([
        SystemMessage(content=RESEARCH_AGENT_PROMPT),
        HumanMessage(
            content=(
                f"{synthesis_instruction}\n\n"
                f"{combined_findings}\n\n"
                f"Tool Reference Summary:\n{reference_text}"
            )
        ),
    ])

    fallback_raw_output = extract_message_content(response)
    fallback_report = clean_markdown_output(fallback_raw_output)
    return fallback_report, fallback_raw_output


def convert_markdown_to_html(markdown_text: str) -> str:
    """Convert markdown into HTML using python-markdown with extra features."""
    html_output = markdown.markdown(
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )
    return normalize_text_characters(html_output)


def create_pdf_report_from_html(html_text: str, subject: str) -> bytes:
    """Generate a PDF from HTML content."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    safe_subject = html.escape(subject).encode("latin-1", "ignore").decode("latin-1")
    safe_html = html_text.encode("latin-1", "ignore").decode("latin-1")
    pdf.write_html(f"<h2>Research Report: {safe_subject}</h2>")
    pdf.write_html(safe_html)
    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", errors="ignore")


def display_report_section(report_state: Dict[str, Any], show_agent_thoughts: bool) -> None:
    """Render the stored research report along with controls and references."""
    if not report_state:
        return

    st.markdown("---")
    st.markdown("## Research Report")
    st.markdown(report_state["report_html"], unsafe_allow_html=True)

    if report_state.get("fallback_used"):
        reason = report_state.get("fallback_reason") or "unknown_reason"
        st.info(f"Displayed report generated via fallback synthesizer (reason: {reason}).")

    pdf_bytes = create_pdf_report_from_html(report_state["report_html"], report_state["query"])

    if show_agent_thoughts and report_state.get("raw_output"):
        with st.expander("Raw Output", expanded=False):
            st.code(report_state["raw_output"], language="markdown")

    safe_query = report_state["query"].replace(" ", "_")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.download_button(
            label="Download Report (MD)",
            data=report_state["report_text"],
            file_name=f"research_report_{safe_query}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"research_report_{safe_query}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col3:
        if st.button("New Search", use_container_width=True):
            st.session_state.pending_reset = True
            st.rerun()

    references = report_state.get("tool_references") or []
    if references:
        st.markdown("---")
        st.markdown("### Sources & Search Queries")
        for ref in references:
            tool_name = ref.get("tool") or "Tool"
            query = ref.get("query") or "N/A"
            st.markdown(f"**{tool_name}** -- query: `{query}`")
            for entry in ref.get("entries", []):
                title = entry.get("title") or "Untitled Source"
                url = entry.get("url") or ""
                if url:
                    st.markdown(f"- [{title}]({url})")
                else:
                    st.markdown(f"- {title}")


class QueryInput(BaseModel):
    """Pydantic schema for structured tool inputs."""

    query: str = Field(..., description="The research subject or topic to investigate.")

# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================
# Wrap our functions as LangChain Tools
# Tools are what the agent can "call" to perform actions
# The LLM reads the tool descriptions and decides which tools to use and when

tools = [
    StructuredTool.from_function(
        name="Search_Web",
        func=search_web,
        description="Useful for searching general information about a person, place, or thing. Use this to get comprehensive background information, facts, and details from the web. This should be your primary tool for gathering information.",
        args_schema=QueryInput,
    ),
    StructuredTool.from_function(
        name="Search_News",
        func=get_news,
        description="Useful for searching the latest news and updates about a person, place, or thing. This should be your secondary tool for gathering information.",
        args_schema=QueryInput,
    ),
    StructuredTool.from_function(
        name="Get_DuckDuckGo_News",
        func=duckduckgo_news,
        description="Useful for finding recent news from various sources via DuckDuckGo. Use this as an alternative news source to complement News API.",
        args_schema=QueryInput,
    ),
]

# ============================================================================
# AGENT PROMPT
# ============================================================================
# This is the system prompt that defines the agent's behavior and output format
# It's passed as a SystemMessage to ensure the LLM follows these instructions
# throughout the entire conversation

RESEARCH_AGENT_PROMPT = """
# ROLE
You are an Expert Research Analyst AI with deep expertise in synthesizing information from multiple sources. You have advanced skills in critical thinking, fact-checking, information synthesis, and presenting complex data in clear, structured formats. You approach research with academic rigor while making findings accessible to general audiences.

# PROCESS
When conducting research on a person, place, or thing, follow this systematic approach:

1. **Initial Analysis**
   - Identify the specific subject and determine what type of entity it is (person, place, organization, concept, etc.)
   - Clarify any ambiguities in the query
   - Determine the key aspects that should be researched

2. **Information Gathering**
   - ALWAYS start with Web Search to find comprehensive background information and facts
   - Use News API Search to find recent developments from major publications
   - Use DuckDuckGo News for additional news coverage and perspectives
   - Cross-reference information across multiple sources to verify accuracy
   - Prioritize authoritative and recent sources

3. **Critical Evaluation**
   - Assess the credibility and recency of information
   - Identify consensus information vs. conflicting reports
   - Note any biases or limitations in available information
   - Flag unverified claims or areas where information is incomplete

4. **Synthesis and Structuring**
   - Organize findings into clear, logical sections
   - Prioritize the most relevant and important information
   - Present a balanced view that includes different perspectives when applicable
   - Use proper attribution for information

5. **Presentation**
   - Begin with a concise executive summary (2-3 sentences)
   - Structure the main content with clear headings and subheadings
   - Use bullet points for clarity and scannability
   - Include key takeaways
   - Flag any areas where information is uncertain

# GOAL
Your goal is to provide a comprehensive, accurate, and well-structured research report that:
- Answers the query thoroughly and objectively
- Presents information from multiple credible sources
- Highlights recent developments and current relevance
- Is easy to read and navigate
- Maintains high standards of accuracy and intellectual honesty

# OUTPUT FORMAT
CRITICAL: Your Final Answer MUST be a well-formatted markdown document. Structure your response EXACTLY as follows:

## Executive Summary
[2-3 sentence overview of the subject]

## Overview
[Comprehensive background information]

## Recent Developments
[Latest news, events, or updates from the past 6-12 months]

## Key Facts and Details
[Important specific information, statistics, dates, etc.]

## Different Perspectives
[If applicable, present multiple viewpoints or controversies]

## Key Takeaways
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]

## References
- [Source name](URL) - include context or query used
- [Repeat for every verified source pulled from the tools]
- [If a tool provided multiple important links, list each one]

## Research Notes
[Any limitations, gaps in information, or areas requiring further investigation]

CRITICAL REMINDERS:
- Your FINAL ANSWER must be ONLY the markdown-formatted report, nothing else
- Do NOT include "Final Answer:" or any prefix - just start with "## Executive Summary"
- You MUST include ALL sections listed above in the exact order shown
- Each section MUST start with its markdown heading (e.g., ## Executive Summary)
- Include blank lines between sections for readability
- Use bullet points (-) for Key Takeaways section
- The References section MUST cite real URLs surfaced by the tools (do not invent or leave blank)
- Format the entire report as a clean, readable markdown document
- When you are ready to provide the final answer, output ONLY the markdown report starting with "## Executive Summary"

# CONSTRAINTS
- Always verify information across multiple sources when possible
- Clearly distinguish between verified facts and claims
- If information conflicts across sources, acknowledge this explicitly
- Never fabricate information - if something is unknown, say so
- Prioritize recent information (< 1 year old) when discussing current events
- Maintain objectivity and avoid editorializing
- If the subject is controversial, present multiple perspectives fairly

# SPECIAL INSTRUCTIONS
- For people: Focus on notable achievements, current activities, background, and public impact
- For places: Cover geography, history, current significance, demographics, and recent developments
- For things/concepts: Explain definition, origin, current usage, significance, and recent innovations
- Mention the specific tools or queries used when listing references when it adds clarity

TOOLS AVAILABLE:
- Web Search: Use for general web information and background (USE THIS FIRST)
- News API Search: Use for recent news from major publications
- DuckDuckGo News: Use for additional news coverage

You MUST use at least Web Search and one news tool to gather comprehensive information before providing your final report.
"""

# ============================================================================
# LANGGRAPH AGENT SETUP
# ============================================================================
# LangGraph is used to create a ReAct (Reasoning + Acting) agent
# The agent can call tools, process results, and make decisions in a loop
# until it has enough information to provide a final answer

# Define the agent's state structure
# The state contains a list of messages (conversation history)
# Annotated[list[BaseMessage], add] means messages are appended, not replaced
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]

# Create a ReAct prompt template
# This template combines the system prompt with tool information
# The agent uses this to understand what tools are available and how to use them
# Note: This prompt template is created but not directly used in the current implementation
# Instead, we pass the RESEARCH_AGENT_PROMPT as a SystemMessage
prompt = PromptTemplate.from_template(f"""{RESEARCH_AGENT_PROMPT}

You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have all the information needed. I will provide my final answer in the exact markdown format specified in the OUTPUT FORMAT section.
Final Answer: [Output ONLY the markdown report starting with "## Executive Summary" - do NOT include "Final Answer:" prefix. Follow the exact structure: Executive Summary, Overview, Recent Developments, Key Facts & Details, Different Perspectives, Key Takeaways, References, Research Notes. Include all sections in order.]

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}""")

# Bind tools to the LLM
# This enables the LLM to "call" our tools (search_web, get_news, etc.)
# When the LLM decides to use a tool, it returns a tool call instead of text
llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# LANGGRAPH NODE FUNCTIONS
# ============================================================================

def should_continue(state: AgentState):
    """
    Determine whether the agent should continue or finish.
    
    This function is called after the agent node to decide the next step:
    - If the last message has tool calls -> continue to tools node
    - If the last message has no tool calls -> end (agent provided final answer)
    
    Args:
        state: The current agent state containing messages
        
    Returns:
        "continue" if tools should be called, "end" if agent is done
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message contains tool calls
    # Tool calls indicate the agent wants to use a tool
    if not last_message.tool_calls:
        return "end"  # No tool calls = agent provided final answer
    return "continue"  # Has tool calls = need to execute tools

def call_model(state: AgentState):
    """
    Call the LLM with the current conversation state.
    
    This is the "agent" node in the LangGraph workflow.
    It:
    1. Ensures the system prompt is included
    2. Invokes the LLM with the conversation history
    3. Returns the LLM's response (which may include tool calls)
    
    Args:
        state: The current agent state containing messages
        
    Returns:
        Updated state with the LLM's response added
    """
    messages = state["messages"]
    
    # Check if system message is already in the messages
    # This is a safeguard - the system message should be added at invocation time
    has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
    
    # Prepend system message with research prompt if not present
    # This ensures the agent always has the instructions, even if something goes wrong
    if not has_system_message:
        system_message = SystemMessage(content=RESEARCH_AGENT_PROMPT)
        messages = [system_message] + messages
    
    # Invoke the LLM with the conversation history
    # The LLM will either:
    # 1. Return a text response (final answer)
    # 2. Return tool calls (request to use tools)
    response = llm_with_tools.invoke(messages)
    
    # Return the response as a new message in the state
    # The "add" annotation in AgentState means this message is appended to the list
    return {"messages": [response]}

# ============================================================================
# LANGGRAPH WORKFLOW CONSTRUCTION
# ============================================================================
# Build the agent workflow as a graph with nodes and edges
# The workflow defines how the agent moves between states

# Create a new StateGraph with our AgentState structure
workflow = StateGraph(AgentState)

# Add nodes to the graph
# Nodes are functions that process the state

# "agent" node: Calls the LLM to generate a response or tool calls
workflow.add_node("agent", call_model)

# "tools" node: Executes tool calls returned by the agent
# ToolNode is a prebuilt LangGraph node that automatically executes tool calls
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Set the entry point - where the workflow starts
workflow.set_entry_point("agent")

# Add conditional edges from the agent node
# After the agent runs, we check if it wants to use tools or is done
workflow.add_conditional_edges(
    "agent",           # From this node
    should_continue,   # Use this function to decide the next step
    {
        "continue": "tools",  # If continue -> go to tools node
        "end": END,           # If end -> finish the workflow
    }
)

# Add an edge from tools back to agent
# After tools execute, we go back to the agent to process the results
# This creates a loop: agent -> tools -> agent -> tools -> ... -> agent -> end
workflow.add_edge("tools", "agent")

# Compile the graph into an executable agent
# MemorySaver enables conversation memory/checkpointing
# This allows the agent to remember context across multiple interactions
agent_executor = workflow.compile(checkpointer=MemorySaver())

# ============================================================================
# STREAMLIT UI SETUP
# ============================================================================
# Streamlit is used to create the web interface
# It automatically handles web server, routing, and state management

# Configure the Streamlit page
st.set_page_config(page_title="Real-Time Research Agent", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .research-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Real-Time Research Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Research any person, place, or thing with AI-powered analysis</p>", unsafe_allow_html=True)

# Initialize persistent UI/session state
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "latest_report" not in st.session_state:
    st.session_state.latest_report = None
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""
if "pending_reset" not in st.session_state:
    st.session_state.pending_reset = False

if st.session_state.pending_reset:
    st.session_state.query_input = ""
    st.session_state.latest_report = None
    st.session_state.selected_query = ""
    st.session_state.pending_reset = False
    reset_tool_references()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    show_agent_thoughts = st.checkbox("Show Agent Reasoning", value=False, help="Display the agent's thought process and tool usage")
    st.markdown("---")
    st.markdown("### Tools Used")
    st.markdown("""
    - **DuckDuckGo Search**: General web information
    - **News API**: Recent news from major publications
    - **DuckDuckGo News**: Additional news coverage
    - **Gemini Flash**: AI analysis and synthesis
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This research agent:
    - Searches the web comprehensively
    - Finds recent news articles
    - Synthesizes information
    - Provides structured reports
    - Cross-references sources
    """)
    st.markdown("---")
    st.markdown("### API Keys Required")
    google_key_status = "Ready" if GOOGLE_API_KEY else "Missing"
    news_key_status = "Ready" if NEWS_API_KEY else "Missing"
    st.markdown(f"- Google API Key: {google_key_status}")
    st.markdown(f"- News API Key: {news_key_status}")
    st.markdown("---")
    st.markdown("### About Report Format")
    st.markdown("""
    Reports are formatted with markdown for human readability:
    - Clear section headings
    - Bullet points for key takeaways
    - Structured sections for easy navigation
    """)
# Main interface
st.markdown("---")

if st.session_state.selected_query:
    st.session_state.query_input = st.session_state.selected_query
    st.session_state.selected_query = ""

query = st.text_input(
    "What would you like to research?",
    placeholder="e.g., Elon Musk, Paris France, Quantum Computing, ChatGPT, etc.",
    help="Enter any person, place, thing, or concept you want to research",
    key="query_input",
)

col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    search_button = st.button("Research", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.pending_reset = True
    st.rerun()

# ============================================================================
# RESEARCH EXECUTION
# ============================================================================
# This section handles the actual research when the user clicks "Research"

if search_button and not query.strip():
    st.warning("Please enter a query to research.")

if search_button and query.strip():
    # Validate API key before proceeding
    if not GOOGLE_API_KEY:
        st.error("Please set your GOOGLE_API_KEY in environment variables")
    else:
        # Set up progress indicators for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing research...")
        progress_bar.progress(10)
        
        # Execute the research in a spinner for visual feedback
        with st.spinner(f"Researching '{query}'..."):
            try:
                # Optional: Show agent reasoning if user enabled it
                if show_agent_thoughts:
                    with st.expander("Agent Reasoning Process", expanded=True):
                        st.info("The agent will use multiple tools to gather comprehensive information...")
                
                status_text.text("Searching the web...")
                progress_bar.progress(30)
                
                # ============================================================
                # AGENT INVOCATION
                # ============================================================
                # This is where the agent is actually executed
                # The agent will:
                # 1. Receive the query and system prompt
                # 2. Decide which tools to use
                # 3. Call tools (search_web, get_news, etc.)
                # 4. Process tool results
                # 5. Repeat until it has enough information
                # 6. Generate the final research report
                
                # Construct the research query with explicit format instructions
                research_query = f"Conduct comprehensive research on: {query}. You MUST provide your final answer as a well-formatted markdown document following the structure specified in the OUTPUT FORMAT section. Your response must be ONLY the markdown-formatted report, nothing else."
                
                # Configuration for the agent execution
                # thread_id allows the agent to maintain conversation context
                # Using "1" as a simple thread ID (in production, use unique IDs per user/session)
                config = {"configurable": {"thread_id": "1"}}
                
                # Create initial messages for the agent
                # SystemMessage: Contains the research prompt with all instructions
                # HumanMessage: Contains the user's research query
                initial_messages = [
                    SystemMessage(content=RESEARCH_AGENT_PROMPT),  # Agent instructions
                    HumanMessage(content=research_query)           # User query
                ]
                
                # Invoke the agent executor
                # This starts the LangGraph workflow:
                # agent -> (tool calls?) -> tools -> agent -> ... -> final answer
                reset_tool_references()
                
                result = agent_executor.invoke(
                    {"messages": initial_messages},
                    config=config
                )
                
                # ============================================================
                # EXTRACT FINAL ANSWER
                # ============================================================
                # The result contains all messages from the conversation
                # We need to find the final answer (the last message without tool calls)
                
                messages = result.get("messages", [])
                raw_output = ""
                finish_reason = None
                
                if messages:
                    # Find the last AIMessage that doesn't have tool calls
                    final_answer = None
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            final_answer = msg
                            break
                
                    # Fallback: if no final answer found, use the last message
                    if final_answer is None:
                        final_answer = messages[-1]
                
                    if hasattr(final_answer, "response_metadata"):
                        finish_reason = final_answer.response_metadata.get("finish_reason")
                
                    raw_output = extract_message_content(final_answer)
                else:
                    # Fallback if no messages found
                    raw_output = str(result)
                
                # Clean and format the markdown output
                report_text = normalize_text_characters(clean_markdown_output(raw_output))
                report_html = convert_markdown_to_html(report_text)
                
                fallback_used = False
                fallback_reason = None
                if (not report_text.strip()) or finish_reason == "MALFORMED_FUNCTION_CALL":
                    fallback_used = True
                    fallback_reason = finish_reason or "empty_output"
                    status_text.text("Recovering from agent error. Re-running synthesis...")
                    progress_bar.progress(60)
                    reset_tool_references()
                    report_text, raw_output = run_direct_research_synthesis(query)
                    report_text = normalize_text_characters(report_text)
                    report_html = convert_markdown_to_html(report_text)
                
                    if not report_text.strip():
                        raise ValueError("Fallback synthesis did not return any content.")
                
                status_text.text("Research complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.latest_report = {
                    "query": query,
                    "report_text": report_text,
                    "report_html": report_html,
                    "raw_output": raw_output,
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "tool_references": copy.deepcopy(CURRENT_TOOL_REFERENCES),
                }
                
            except Exception as e:
                # Error handling: Display user-friendly error messages
                st.error(f"An error occurred: {str(e)}")
                st.info("Try rephrasing your query or check your API keys.")

latest_report = st.session_state.get("latest_report")
if latest_report:
    display_report_section(latest_report, show_agent_thoughts)

# Example queries
st.markdown("---")
with st.expander("Example Research Topics", expanded=not search_button):
    st.markdown("Click any example to start researching:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**People**")
        people = ["Elon Musk", "Taylor Swift", "Satya Nadella", "Sam Altman"]
        for person in people:
            if st.button(person, key=f"person_{person}", use_container_width=True):
                st.session_state.selected_query = person
                st.rerun()
    
    with col2:
        st.markdown("**Places**")
        places = ["Tokyo Japan", "Grand Canyon", "CERN", "Mars"]
        for place in places:
            if st.button(place, key=f"place_{place}", use_container_width=True):
                st.session_state.selected_query = place
                st.rerun()
    
    with col3:
        st.markdown("**Things/Concepts**")
        things = ["Quantum Computing", "CRISPR", "ChatGPT", "Blockchain"]
        for thing in things:
            if st.button(thing, key=f"thing_{thing}", use_container_width=True):
                st.session_state.selected_query = thing
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with dedication using LangChain, Gemini, DuckDuckGo, and Streamlit<br>"
    "No Google Custom Search API required!"
    "</div>",
    unsafe_allow_html=True
)


