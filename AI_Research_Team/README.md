# AI Research & Newsletter Generator

A multi-agent system built with **CrewAI** that automatically researches a topic, analyzes the findings, and writes a ready-to-publish newsletter — with built-in safeguards against hallucinated content.

Given a topic (e.g. *"Chinese AI models"*, *"Open Source AI Models"*), the crew searches for real, recent news, evaluates their impact, and produces a polished Markdown newsletter, all through a simple Streamlit interface.

## How it works

The project uses three specialized AI agents that collaborate in sequence:

1. **Researcher** — searches the web for the most recent, relevant news on the given topic using the [Serper.dev](https://serper.dev) Google News API. Only real, verifiable results (with title, date, source, and URL) are accepted; the agent is explicitly instructed not to invent information.
2. **Analyst** — reviews the raw research and evaluates the real-world technical and business impact of each finding, producing structured insights and comparison tables.
3. **Writer** — turns the analysis into a polished, engaging newsletter formatted in Markdown.

### Anti-hallucination safeguards

Since LLMs can be tempted to fabricate plausible-sounding content when search results are weak or missing, the pipeline includes several layers of protection:

- **Pre-check before running the crew**: the app searches for news in the last 7 days; if none are found, it automatically widens the search to the last 30 days before giving up.
- **Guardrail on the research task**: CrewAI automatically rejects and retries any research output that doesn't contain real URLs.
- **Explicit stop conditions**: every agent is instructed to output a clear "insufficient data" message rather than inventing facts, and the app detects this message and shows an error instead of a fabricated newsletter.

## Tech stack

- [CrewAI](https://www.crewai.com/) — multi-agent orchestration
- [DeepSeek](https://www.deepseek.com/) (`deepseek-chat`) — the LLM powering each agent
- [Serper.dev](https://serper.dev) — Google News search API (real-time, rate-limit friendly)
- [Streamlit](https://streamlit.io/) — web interface
- Python 3.10+

## Project structure

```
.
├── app.py              # Main application: Streamlit UI, agents, tasks, and crew definition
├── .env                # API keys (not committed to git)
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:

```
DEEPSEEK_API_KEY=your-deepseek-key-here
SERPER_API_KEY=your-serper-key-here
```

- Get a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com)
- Get a free Serper API key (2,500 free searches) from [serper.dev](https://serper.dev)

3. Run the app:

```bash
streamlit run app.py
```

4. Enter a topic in the input field and click **Run Crew**. The generated newsletter will be displayed in the app and saved locally as a `.md` file.

## Example output

Given a topic like *"AI Agents"*, the crew produces a Markdown newsletter with:

- A catchy title and engaging introduction
- A dedicated section for each of the 3 most significant recent news items, including technical details, business impact analysis, comparison tables, and source links
- A brief closing thought

## Notes & limitations

- News availability depends on what Serper/Google News indexes; very niche or non-English topics may return fewer results.
- The free Serper tier includes 2,500 searches total (not renewed monthly) — monitor usage if testing frequently.
- The system is designed to fail gracefully: if no real news can be found, it will clearly say so instead of generating fabricated content.

## License

MIT — feel free to use, modify, and share.