# 🔍 Real-Time Research Agent

An intelligent AI-powered research agent that conducts comprehensive research on any topic using real-time web search and news aggregation. Built with LangChain, LangGraph, Google Gemini, and Streamlit, this agent synthesizes information from multiple sources and presents structured, well-formatted research reports.

## ✨ Features

- **Multi-Source Research**: Aggregates information from DuckDuckGo web search, News API, and DuckDuckGo News
- **AI-Powered Analysis**: Uses Google Gemini 2.5 Flash to analyze and synthesize information from multiple sources
- **Structured Reports**: Generates well-formatted markdown reports with:
  - Executive Summary
  - Overview
  - Recent Developments
  - Key Facts & Details
  - Different Perspectives
  - Key Takeaways
  - Research Notes
- **Real-Time Data**: Fetches up-to-date information from the web and recent news articles
- **Interactive UI**: Clean, user-friendly Streamlit interface with progress indicators
- **Export Functionality**: Download research reports as markdown files
- **Agent Reasoning**: Optional view of agent's thought process and tool usage

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash (via LangChain)
- **Agent Framework**: LangChain + LangGraph
- **Search Tools**: 
  - DuckDuckGo Search (web search)
  - DuckDuckGo News (news aggregation)
  - News API (news from major publications)
- **Language**: Python 3.x

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini)
- News API Key (optional, for News API access)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Real-Time-Research-Agent.git
   cd Real-Time-Research-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   NEWS_API_KEY=your_news_api_key_here
   ```
   
   **Getting API Keys:**
   - **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **News API Key**: Get from [News API](https://newsapi.org/) (free tier available)

## 🎯 Usage

1. **Start the Streamlit application**
   ```bash
   streamlit run agent.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

3. **Research a topic**
   - Enter any person, place, thing, or concept in the search box
   - Click "🔍 Research" to start
   - Wait for the agent to gather information and generate the report

4. **View and download results**
   - The research report will be displayed in a structured markdown format
   - Use the "📥 Download Report (MD)" button to save the report
   - Toggle "Show Agent Reasoning" to see the agent's thought process

## 📊 Example Research Topics

The agent can research a wide variety of topics:

- **People**: Elon Musk, Taylor Swift, Satya Nadella, etc.
- **Places**: Tokyo Japan, Grand Canyon, CERN, etc.
- **Concepts**: Quantum Computing, CRISPR, ChatGPT, Blockchain, etc.

## 🔧 How It Works

1. **Query Processing**: The user enters a research query
2. **Agent Initialization**: The agent receives a system prompt with detailed instructions
3. **Information Gathering**: The agent uses multiple tools:
   - Web Search (DuckDuckGo) for general information
   - News API for recent news from major publications
   - DuckDuckGo News for additional news coverage
4. **Analysis & Synthesis**: The LLM analyzes all gathered information
5. **Report Generation**: The agent creates a structured markdown report following a predefined format
6. **Output**: The report is displayed and can be downloaded

## 📁 Project Structure

```
Real-Time-Research-Agent/
├── agent.py              # Main application file
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md            # This file
```

## 🎨 Report Format

Each research report follows a consistent structure:

1. **📋 Executive Summary**: 2-3 sentence overview
2. **📖 Overview**: Comprehensive background information
3. **📰 Recent Developments**: Latest news and updates (past 6-12 months)
4. **📊 Key Facts & Details**: Important statistics, dates, and specific information
5. **💭 Different Perspectives**: Multiple viewpoints or controversies (if applicable)
6. **✨ Key Takeaways**: Bullet-point summary
7. **📝 Research Notes**: Limitations, gaps, or areas requiring further investigation

## ⚙️ Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required - Your Google Gemini API key
- `NEWS_API_KEY`: Optional - Your News API key (for News API access)

### Agent Settings

The agent can be configured in `agent.py`:
- **LLM Model**: Currently set to `gemini-2.5-flash` (can be changed)
- **Search Results**: Number of results per search (default: 10)
- **Report Format**: Customizable in `RESEARCH_AGENT_PROMPT`

## 🔍 Tools Used

- **Search_Web**: DuckDuckGo web search for general information
- **Search_News**: News API for recent news from major publications
- **Get_DuckDuckGo_News**: DuckDuckGo News for additional news coverage

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Errors**: Ensure your `.env` file is properly configured with valid API keys

3. **News API Limits**: Free tier has rate limits. If you hit limits, the agent will continue using DuckDuckGo search

4. **Streamlit Not Starting**: Make sure Streamlit is installed and Python path is correct
   ```bash
   pip install streamlit
   streamlit run agent.py
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Google Gemini](https://deepmind.google/technologies/gemini/) - AI model
- [Streamlit](https://streamlit.io/) - Web framework
- [DuckDuckGo](https://duckduckgo.com/) - Privacy-focused search
- [News API](https://newsapi.org/) - News aggregation

## 📧 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Built with ❤️ using LangChain, Gemini, DuckDuckGo, and Streamlit**
