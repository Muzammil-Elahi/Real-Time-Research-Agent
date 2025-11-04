import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from operator import add
from duckduckgo_search import DDGS
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
# get api keys and initialize llm
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# create functions for tools

# Custom DuckDuckGo search function for more control
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo for comprehensive information"""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=10)
        
        if not results:
            return "No web results found."
        
        search_summary = "Web Search Results:\n\n"
        for idx, result in enumerate(results, 1):
            search_summary += f"{idx}. Title: {result['title']}\n"
            search_summary += f"   URL: {result['href']}\n"
            search_summary += f"   Snippet: {result['body']}\n\n"
        
        return search_summary
    except Exception as e:
        return f"Error searching web: {str(e)}"

# News search function
def get_news(query: str) -> str:
    """Search for recent news articles about the query"""
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=10
        )
        
        if articles['totalResults'] == 0:
            return "No recent news articles found."
        
        news_summary = "Recent News Articles:\n\n"
        for idx, article in enumerate(articles['articles'][:10], 1):
            news_summary += f"{idx}. Title: {article['title']}\n"
            news_summary += f"   Source: {article['source']['name']}\n"
            news_summary += f"   Date: {article['publishedAt']}\n"
            news_summary += f"   Description: {article.get('description', 'No description')}\n"
            news_summary += f"   URL: {article['url']}\n\n"
        
        return news_summary
    except Exception as e:
        return f"Error fetching news: {str(e)}"

# DuckDuckGo News search function
def duckduckgo_news(query: str) -> str:
    """Search for recent news using DuckDuckGo News"""
    try:
        ddgs = DDGS()
        news_results = ddgs.news(query, max_results=10)
        
        if not news_results:
            return "No news results found."
        
        news_summary = "DuckDuckGo News Results:\n\n"
        for idx, article in enumerate(news_results, 1):
            news_summary += f"{idx}. Title: {article['title']}\n"
            news_summary += f"   Source: {article['source']}\n"
            news_summary += f"   Date: {article['date']}\n"
            news_summary += f"   Snippet: {article['body']}\n"
            news_summary += f"   URL: {article['url']}\n\n"
        
        return news_summary
    except Exception as e:
        return f"Error fetching DDG news: {str(e)}"

# Clean and format markdown output
def clean_markdown_output(raw_output: str) -> str:
    """
    Clean the agent output to ensure it's properly formatted markdown.
    Removes any code blocks or unwanted formatting.
    """
    # Remove markdown code blocks if present
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```markdown"):
        cleaned_output = cleaned_output[11:]  # Remove ```markdown
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]  # Remove ```
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]  # Remove trailing ```
    
    # Remove "Final Answer:" prefix if present
    if cleaned_output.startswith("Final Answer:"):
        cleaned_output = cleaned_output[13:].strip()
    elif cleaned_output.startswith("Final Answer"):
        cleaned_output = cleaned_output[12:].strip()
    
    # Remove any leading text before the first markdown heading
    # Find the first markdown heading (##)
    first_heading = cleaned_output.find("##")
    if first_heading > 0 and first_heading < 100:  # Only remove if heading is near the start
        cleaned_output = cleaned_output[first_heading:].strip()
    
    # Strip leading/trailing whitespace but preserve internal formatting
    return cleaned_output.strip()

# create tools given the functions
tools = [
    Tool(
        name="Search_Web",
        func=search_web,
        description="Useful for searching general information about a person, place, or thing. Use this to get comprehensive background information, facts, and details from the web. This should be your primary tool for gathering information."
        ),
    Tool(
        name="Search_News",
        func=get_news,
        description="Useful for searching the latest news and updates about a person, place, or thing. Use this to get the latest news and updates about a person, place, or thing. This should be your secondary tool for gathering information.",
    ),
    Tool(
        name="Get_DuckDuckGo_News",
        func=duckduckgo_news,
        description="Useful for finding recent news from various sources via DuckDuckGo. Use this as an alternative news source to complement News API. Good for diverse perspectives and additional coverage."
    )
]

# create prompt for agent
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

## 📋 Executive Summary

[2-3 sentence overview of the subject]

## 📖 Overview

[Comprehensive background information]

## 📰 Recent Developments

[Latest news, events, or updates from the past 6-12 months]

## 📊 Key Facts & Details

[Important specific information, statistics, dates, etc.]

## 💭 Different Perspectives

[If applicable, present multiple viewpoints or controversies]

## ✨ Key Takeaways

- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]

## 📝 Research Notes

[Any limitations, gaps in information, or areas requiring further investigation]

CRITICAL REMINDERS: 
- Your FINAL ANSWER must be ONLY the markdown-formatted report, nothing else
- Do NOT include "Final Answer:" or any prefix - just start with "## 📋 Executive Summary"
- You MUST include ALL sections listed above in the exact order shown
- Each section MUST start with its markdown heading (e.g., ## 📋 Executive Summary)
- Include blank lines between sections for readability
- Use bullet points (-) for Key Takeaways section
- Format the entire report as a clean, readable markdown document
- When you are ready to provide the final answer, output ONLY the markdown report starting with "## 📋 Executive Summary"

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

TOOLS AVAILABLE:
- Web Search: Use for general web information and background (USE THIS FIRST)
- News API Search: Use for recent news from major publications
- DuckDuckGo News: Use for additional news coverage

You MUST use at least Web Search and one news tool to gather comprehensive information before providing your final report.
"""

# Create agent using LangGraph (compatible with LangChain 1.0+)
# Define the state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]

# Create the react prompt template
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
Final Answer: [Output ONLY the markdown report starting with "## 📋 Executive Summary" - do NOT include "Final Answer:" prefix. Follow the exact structure: Executive Summary, Overview, Recent Developments, Key Facts & Details, Different Perspectives, Key Takeaways, Research Notes. Include all sections in order.]

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}""")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Create the agent graph
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there are, we continue
    return "continue"

def call_model(state: AgentState):
    messages = state["messages"]
    # Check if system message is already in the messages
    has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
    
    # Prepend system message with research prompt if not present
    if not has_system_message:
        system_message = SystemMessage(content=RESEARCH_AGENT_PROMPT)
        messages = [system_message] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the graph
workflow = StateGraph(AgentState)

# Add the node, we'll call this one "agent"
workflow.add_node("agent", call_model)

# Add the tool node
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

# Add an edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
agent_executor = workflow.compile(checkpointer=MemorySaver())

# create frontend
# Streamlit UI
st.set_page_config(page_title="Real-Time Research Agent", page_icon="🔍", layout="wide")

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

st.markdown("<h1 class='main-header'>🔍 Real-Time Research Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Research any person, place, or thing with AI-powered analysis</p>", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    show_agent_thoughts = st.checkbox("Show Agent Reasoning", value=False, help="Display the agent's thought process and tool usage")
    
    st.markdown("---")
    st.markdown("### 🔧 Tools Used")
    st.markdown("""
    - **DuckDuckGo Search**: General web information
    - **News API**: Recent news from major publications
    - **DuckDuckGo News**: Additional news coverage
    - **Gemini flash**: AI analysis and synthesis
    """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This research agent:
    - Searches the web comprehensively
    - Finds recent news articles
    - Synthesizes information
    - Provides structured reports
    - Cross-references sources
    """)
    
    st.markdown("---")
    st.markdown("### 🔑 API Keys Required")
    google_key_status = "✅" if GOOGLE_API_KEY else "❌"
    news_key_status = "✅" if NEWS_API_KEY else "❌"
    st.markdown(f"- Google API Key: {google_key_status}")
    st.markdown(f"- News API Key: {news_key_status}")
    
    st.markdown("---")
    st.markdown("### 📝 About Report Format")
    st.markdown("""
    Reports are formatted with markdown for human readability:
    - Clear section headings
    - Bullet points for key takeaways
    - Structured sections for easy navigation
    """)

# Main interface
st.markdown("---")

query = st.text_input(
    "🔎 What would you like to research?",
    placeholder="e.g., Elon Musk, Paris France, Quantum Computing, ChatGPT, etc.",
    help="Enter any person, place, thing, or concept you want to research"
)

col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    search_button = st.button("🔍 Research", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Research execution
if search_button and query:
    if not GOOGLE_API_KEY:
        st.error("⚠️ Please set your GOOGLE_API_KEY in environment variables")
    else:
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Initializing research...")
        progress_bar.progress(10)
        
        with st.spinner(f"🔍 Researching '{query}'..."):
            try:
                # Create agent thought container
                if show_agent_thoughts:
                    with st.expander("🧠 Agent Reasoning Process", expanded=True):
                        st.info("The agent will use multiple tools to gather comprehensive information...")
                
                status_text.text("🌐 Searching the web...")
                progress_bar.progress(30)
                
                # Run the agent with custom research instructions
                research_query = f"Conduct comprehensive research on: {query}. You MUST provide your final answer as a well-formatted markdown document following the structure specified in the OUTPUT FORMAT section. Your response must be ONLY the markdown-formatted report, nothing else."
                config = {"configurable": {"thread_id": "1"}}
                # Add system message with research prompt to initial state
                initial_messages = [
                    SystemMessage(content=RESEARCH_AGENT_PROMPT),
                    HumanMessage(content=research_query)
                ]
                result = agent_executor.invoke(
                    {"messages": initial_messages},
                    config=config
                )
                # Extract the final message from the agent
                messages = result.get("messages", [])
                raw_output = ""
                if messages:
                    # Find the last AIMessage that doesn't have tool calls (the final answer)
                    final_answer = None
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            final_answer = msg
                            break
                    
                    # If no final answer found, use the last message
                    if final_answer is None:
                        final_answer = messages[-1]
                    
                    if hasattr(final_answer, 'content'):
                        content = final_answer.content
                        # Convert to string if content is a list
                        if isinstance(content, list):
                            raw_output = "\n".join(str(item) for item in content)
                        else:
                            raw_output = str(content)
                    else:
                        raw_output = str(final_answer)
                else:
                    raw_output = str(result)
                
                # Clean and format the markdown output
                report_text = clean_markdown_output(raw_output)
                
                status_text.text("✅ Research complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Research Report")
                
                # Display the report text with proper markdown rendering
                # The text should already contain markdown formatting (headings, bullet points, etc.)
                st.markdown(report_text)
                
                # Display the raw output (for debugging/verification)
                if show_agent_thoughts:
                    with st.expander("📋 Raw Output", expanded=False):
                        st.code(raw_output, language="markdown")
                
                # Action buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Download as Markdown
                    st.download_button(
                        label="📥 Download Report (MD)",
                        data=report_text,
                        file_name=f"research_report_{query.replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 New Search", use_container_width=True):
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Try rephrasing your query or check your API keys.")

# Example queries
st.markdown("---")
with st.expander("💡 Example Research Topics", expanded=not search_button):
    st.markdown("Click any example to start researching:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 People**")
        people = ["Elon Musk", "Taylor Swift", "Satya Nadella", "Sam Altman"]
        for person in people:
            if st.button(person, key=f"person_{person}", use_container_width=True):
                st.session_state.selected_query = person
                st.rerun()
    
    with col2:
        st.markdown("**🌍 Places**")
        places = ["Tokyo Japan", "Grand Canyon", "CERN", "Mars"]
        for place in places:
            if st.button(place, key=f"place_{place}", use_container_width=True):
                st.session_state.selected_query = place
                st.rerun()
    
    with col3:
        st.markdown("**💡 Things/Concepts**")
        things = ["Quantum Computing", "CRISPR", "ChatGPT", "Blockchain"]
        for thing in things:
            if st.button(thing, key=f"thing_{thing}", use_container_width=True):
                st.session_state.selected_query = thing
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with ❤️ using LangChain, Gemini, DuckDuckGo, and Streamlit<br>"
    "No Google Custom Search API required!"
    "</div>",
    unsafe_allow_html=True
)