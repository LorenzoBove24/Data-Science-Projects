import os
import re
import requests
from datetime import datetime, date
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="AI Research Team", page_icon="🤖", layout="wide")
st.title("AI Research & Newsletter Generator")
st.markdown("Enter a specific topic to run the multi-agent editorial crew and generate a professional newsletter.")

topic_input = st.text_input("🎯Research Topic", placeholder="e.g. Open Source AI Models, AI Agents, ...")
run_button = st.button("🔍Run Crew")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_NEWS_URL = "https://google.serper.dev/news"

_news_cache = {}

def fetch_news_serper(query: str, timeframe: str = "qdr:w", max_results: int = 10, retries: int = 2):
    cache_key = (query.lower().strip(), timeframe)
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "us", "hl": "en", "tbs": timeframe, "num": max_results}

    for attempt in range(retries):
        try:
            response = requests.post(SERPER_NEWS_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            results = data.get("news", [])
            _news_cache[cache_key] = results
            return results
        except requests.exceptions.RequestException as e:
            print(f"[fetch_news_serper] Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                return []
    return []


if run_button:
    if not topic_input.strip():
        st.warning("Please enter a topic before running the crew.")
        st.stop()

    if not SERPER_API_KEY:
        st.error("⚠️ SERPER_API_KEY not found. Add it to your .env file.")
        st.stop()

    # --- Pre-check with fallback week -> month ---
    with st.spinner("Checking for available recent news..."):
        active_timeframe = "qdr:w"
        preview_results = fetch_news_serper(topic_input, timeframe="qdr:w", max_results=5)

        if not preview_results:
            st.info("No news found in the last 7 days. Widening search to the last month...")
            preview_results = fetch_news_serper(topic_input, timeframe="qdr:m", max_results=5)
            active_timeframe = "qdr:m"

        if not preview_results:
            st.error("⚠️ No real, verifiable news found for this topic in the last month. Try a broader topic.")
            st.stop()

        st.success(
            f"Found {len(preview_results)} candidate news items "
            f"({'last 7 days' if active_timeframe == 'qdr:w' else 'last 30 days'}). Proceeding..."
        )

    with st.spinner("Crew execution in progress... Please wait."):
        llm_deepseek = LLM(
            model="openai/deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        today_str = date.today().strftime("%Y-%m-%d")

        @tool("Serper News Search")
        def search_tool(query: str) -> str:
            """
            Use this tool to search for the most recent, real news about a given topic
            via Google News (through Serper.dev). Returns title, date, source, and URL.
            If it returns 'No recent news found', do not invent a substitute answer.
            """
            results = fetch_news_serper(query, timeframe=active_timeframe, max_results=10)

            if not results:
                return "No recent news found for this query. Do not invent or fabricate any substitute information."

            formatted = []
            for r in results:
                formatted.append(
                    f"Title: {r.get('title')}\n"
                    f"Date: {r.get('date')}\n"
                    f"Source: {r.get('source')}\n"
                    f"URL: {r.get('link')}\n"
                    f"Excerpt: {r.get('snippet')}\n"
                )
            return "\n---\n".join(formatted)

        research_agent = Agent(
            role="Senior AI & Data Science News Hunter",
            goal=(
                f"Scour the web to find the most recent, impactful, and factual news, breakthroughs, "
                f"or releases about {topic_input}, using ONLY real results returned by the search tool."
            ),
            backstory=(
                "You are an elite AI & Data Science Researcher. You have a strict filtering process: you "
                "completely ignore amateur blogs, clickbait, or unverified rumors. You only prioritize primary "
                "sources and authoritative tech news outlets. Your research is the foundation for critical "
                "business decisions, so accuracy is non-negotiable.\n\n"
                f"TODAY'S DATE IS {today_str}. You must treat this as ground truth.\n\n"
                "ABSOLUTE RULES:\n"
                "1) You MUST use the 'Serper News Search' tool. Never answer from memory.\n"
                "2) You MUST NOT invent, hallucinate, or fabricate any news, names, statistics, quotes, or URLs.\n"
                "3) If the tool returns 'No recent news found', try a different, more specific query. If you "
                "still find nothing real, output exactly: 'I could not find recent verified news.' and stop.\n"
                "4) Never write placeholder sources like '[Hypothetical Blog]' — that is a sign you are "
                "inventing content, which is forbidden."
            ),
            llm=llm_deepseek,
            tools=[search_tool],
            verbose=True,
            max_iter=5
        )

        analyst_agent = Agent(
            role="Lead AI & Data Science Analyst",
            goal="Analyze the raw information provided by the researcher, identify the underlying trends, and evaluate the real-world technical and business impact of these news.",
            backstory=(
                "You are a seasoned AI & Data Science Analyst who distills complex technical information into "
                "actionable insights, answering 'Why does this matter?'. You use Markdown tables and structured "
                "comparisons.\n\n"
                "ABSOLUTE RULE: Only analyze information explicitly present in the Researcher's output. If it "
                "states no verified news was found, output exactly: 'Insufficient verified data to proceed with "
                "analysis.' and stop. Never invent statistics or company names not present in the dossier."
            ),
            llm=llm_deepseek,
            verbose=True
        )

        writer_agent = Agent(
            role="Senior AI & Data Science Technical Writer",
            goal="Transform the analyzed insights into a compelling, well-structured, and engaging newsletter article that highlights the most important developments in the field.",
            backstory=(
                "You are a professional Technical Writer who turns deep analysis into clear, engaging "
                "newsletters, using Markdown headings and bullet points.\n\n"
                "ABSOLUTE RULE: Only use facts, names, and URLs present in upstream outputs. If upstream "
                "outputs indicate insufficient data, output exactly: 'Insufficient verified data to generate a "
                "newsletter for this topic.'"
            ),
            llm=llm_deepseek,
            verbose=True
        )

        def validate_research_output(output):
            text = output.raw if hasattr(output, "raw") else str(output)
            if "could not find recent verified news" in text.lower():
                return (True, output)
            has_url = bool(re.search(r"https?://\S+", text))
            if not has_url:
                return (False, "Output contains no real URLs from the search tool — likely hallucinated.")
            return (True, output)

        research_task = Task(
            description=(
                f"Today's date is {today_str}. Use the 'Serper News Search' tool to identify the 3 most "
                f"significant developments about {topic_input} from real, verifiable sources. Discard any "
                f"result without a clear date and URL."
            ),
            expected_output=(
                "A research dossier with the 3 most important, verifiably real news items, each with title, "
                "date, URL, technical details, and quotes if available. If nothing real was found, output "
                "exactly: 'I could not find recent verified news.'"
            ),
            agent=research_agent,
            guardrail=validate_research_output,
            max_retries=2
        )

        analysis_task = Task(
            description=(
                "Review the research dossier. If it states no verified news was found, stop and report that. "
                "Otherwise, perform a deep critical analysis of each item's real-world impact and business "
                "implications."
            ),
            expected_output=(
                "An analytical brief with 'Why it matters', a comparison table, and a forecast for each topic, "
                "or 'Insufficient verified data to proceed with analysis.' if applicable."
            ),
            agent=analyst_agent
        )
        os.makedirs("newsletters", exist_ok=True)
        file_name = f"newsletters/newsletter_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"

        writer_task = Task(
            description=(
                "Synthesize the Researcher's and Analyst's outputs into an engaging newsletter, using only real "
                "facts and URLs. If either upstream output indicates insufficient data, do not fabricate content."
            ),
            expected_output=(
                "A complete Markdown newsletter with title, intro, one section per news item (facts, analysis, "
                "table, date, URL), and a conclusion. Or exactly: 'Insufficient verified data to generate a "
                "newsletter for this topic.' if applicable."
            ),
            agent=writer_agent,
            output_file=file_name
        )

        crew = Crew(
            agents=[research_agent, analyst_agent, writer_agent],
            tasks=[research_task, analysis_task, writer_task],
            verbose=True
        )

        final_report = crew.kickoff()
        report_text = final_report.raw if hasattr(final_report, "raw") else str(final_report)

        if "insufficient verified data" in report_text.lower() or "could not find recent verified news" in report_text.lower():
            st.error("⚠️ The crew could not find enough real, verified information for this topic.")
        else:
            st.success(f"📄 Newsletter successfully generated and saved to {file_name}!")
            st.markdown(report_text)