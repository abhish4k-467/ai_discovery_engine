import os
import requests

from pydantic_ai import Agent
from dotenv import load_dotenv
from doc_processor import DocumentProcessor
from vector_db import VectorDB

load_dotenv()

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

rag_agent = Agent(
    "groq:openai/gpt-oss-20b",
    system_prompt=(
        "You are an expert IT and Startup Consultant Assistant built for 'The Launch Engine' (https://thelaunchengine.com/). "
        "Your domain is focused on IT consulting, software development strategies (web/mobile), cloud solutions (AWS), "
        "startup advisory, tech stack assessments, and digital transformation. "
        "If a user asks a general or off-topic question, warmly greet them as a consultant of The Launch Engine, "
        "and clearly explain that this internal AI tool is designed to assist with IT strategy, app development planning, "
        "cloud architecture, and startup advisory. Ask how you can support their client deliverables today. "
        "Use the provided document context to answer questions. If the local context lacks the answer, "
        "use the web_search tool to find the latest technological and strategic information online.\n"
        "IMPORTANT FORMATTING RULES: Format your output cleanly in Markdown. Do NOT use HTML tags like <br>. "
        "When mentioning currency or tech specifications, ensure dollar signs are escaped (\\$) so they aren't parsed as LaTeX."
    )
)

@rag_agent.tool_plain
def web_search(query: str) -> str:
    """Search the web using Tavily for up-to-date information. Use this if the provided context is insufficient.
    
    Args:
        query: The search query string.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return "Web search failed: query is empty."

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found in environment."

    max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": api_key,
                "query": cleaned_query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": max_results,
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer")
        if answer:
            return f"Tavily Answer: {answer}"
            
        results = data.get("results", [])
        if not results:
            return "No web results found."
            
        snippets = "\n\n".join(
            [
                f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url')}\nContent: {r.get('content', '')[:600]}"
                for r in results[:3]
            ]
        )
        return f"Web Search Results:\n{snippets}"
        
    except Exception as e:
        return f"Web search failed: {str(e)}"

class RAGSystem:
    def __init__(self):
        load_dotenv()
        self.doc_processor = DocumentProcessor()
        self.vector_db = VectorDB()
        self.agent = rag_agent

        # Check for API key
        if not os.getenv("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY not found. Answer generation will fail.")
            self.agent = None

    @staticmethod
    def integration_status() -> dict:
        """Returns whether required external integrations are configured."""
        return {
            "groq_configured": bool(os.getenv("GROQ_API_KEY")),
            "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        }

    def ingest_file(self, file_path: str) -> None:
        print(f"Ingesting file: {file_path}")
        if file_path.lower().endswith(".pdf"):
            chunks = self.doc_processor.process_pdf(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return

        if chunks:
            self.vector_db.upsert(chunks)

    def ingest_url(self, url: str) -> None:
        print(f"Ingesting URL: {url}")
        chunks = self.doc_processor.process_url(url)
        if chunks:
            self.vector_db.upsert(chunks)

    def query(self, question: str) -> str:
        # 1. Search VectorDB
        try:
            results = self.vector_db.search(question)
        except Exception as e:
            return f"Error retrieving context: {e}"

        if not results:
            context = "No local documents found."
        else:
            context = "\n\n".join([f"Source: {r['source']}\nText: {r['text']}" for r in results])
            
        # 2. Construct Prompt
        tavily_status = (
            "Tavily web search is available."
            if os.getenv("TAVILY_API_KEY")
            else "Tavily web search is NOT available because TAVILY_API_KEY is not configured."
        )
        prompt = (
            f"Local Document Context:\n{context}\n\n"
            f"Integration Status: {tavily_status}\n\n"
            f"User Question: {question}\n\n"
            "Answer based on the context provided, or use the web_search tool if needed and available."
        )

        # 3. Generate Answer
        if self.agent is None:
            if not results:
               return "Generation is unavailable (GROQ_API_KEY missing) and no documents were found."
            snippets = "\n\n".join([f"- Source: {r.get('source')}\n  {r.get('text', '')[:280]}" for r in results[:3]])
            return f"Generation is unavailable because GROQ_API_KEY is not configured.\n\nRelevant context:\n\n{snippets}"

        try:
            result = self.agent.run_sync(prompt)
            return result.output
        except Exception as e:
            if not results:
                return f"LLM error: {e}"
            snippets = "\n\n".join([f"- Source: {r.get('source')}\n  {r.get('text', '')[:280]}" for r in results[:3]])
            return f"I couldn't reach the language model service, but I found relevant context:\n\n{snippets}\n\nLLM error: {e}"

    def generate_research_report(self, topic: str) -> str:
        prompt = f"""You are an expert researcher. Please generate a deep-dive research report on the following topic: {topic}. 
Use the web_search tool to gather up-to-date information before writing the report. 
Structure the report appropriately for consultants with an executive summary, main body, and conclusions."""
        
        if self.agent is None:
            return f"Generation is unavailable because GROQ_API_KEY is not configured. Unable to generate report for {topic}."

        try:
            result = self.agent.run_sync(prompt)
            return result.output
        except Exception as e:
            return f"Error generating research report: {e}"
