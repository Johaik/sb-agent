from typing import List
from .base import Agent
from ..llm.base import LLMProvider
from ..tools.tavily_tool import TavilyTool
from ..tools.rag_tool import RAGSearchTool

class EnricherAgent(Agent):
    def __init__(self, llm: LLMProvider):
        super().__init__(
            name="Enricher",
            instructions="""You are an idea enrichment expert. 
            Your goal is to take a brief research idea or event string and expand it into a detailed, comprehensive description. 
            Identify key aspects that need to be researched, potential angles, and context.
            Output ONLY the enriched description text.""",
            llm=llm
        )

class PlannerAgent(Agent):
    def __init__(self, llm: LLMProvider):
        super().__init__(
            name="Planner",
            instructions="""You are a research planner.
            Given a detailed research description, break it down into specific, actionable research tasks.
            Return the tasks as a JSON list of strings, e.g., ["Task 1", "Task 2"].
            Do not include any other text, just the JSON array.""",
            llm=llm
        )

class CriticAgent(Agent):
    def __init__(self, llm: LLMProvider):
        super().__init__(
            name="Critic",
            instructions="""You are a research quality assurance expert.
            Your job is to evaluate a Research Task and its Result.
            
            Determine if the result comprehensively answers the task description.
            Check for:
            - Completeness
            - Relevance
            - Depth
            
            Output strictly valid JSON with the following structure:
            {
              "approved": boolean,
              "feedback": "string explaining what is missing or why it is approved"
            }
            Do not output any markdown or other text.""",
            llm=llm
        )

class ResearcherAgent(Agent):
    def __init__(self, llm: LLMProvider):
        # Tools
        tavily = TavilyTool()
        rag = RAGSearchTool(llm)
        
        super().__init__(
            name="Researcher",
            instructions="""You are a thorough research assistant. 
            Your goal is to complete the assigned research task using available tools.
            
            Process:
            1. Search for information using Tavily (web) or RAG (internal DB).
            2. Analyze the findings.
            3. Critique: Do you have enough info? Is it accurate?
            4. If needed, search again with refined queries.
            5. When satisfied, provide a comprehensive answer to the task.
            
            Use the tools as many times as needed within the turn limit.""",
            llm=llm,
            tools=[tavily, rag]
        )
    
    def run_with_feedback(self, task: str, feedback: str = None) -> str:
        prompt = task
        if feedback:
            prompt = f"Task: {task}\n\nPREVIOUS FEEDBACK (Must be addressed): {feedback}\n\nPlease improve the research based on this feedback."
        return self.run(prompt)

class ReporterAgent(Agent):
    def __init__(self, llm: LLMProvider):
        super().__init__(
            name="Reporter",
            instructions="""You are a research reporter.
            You will receive a set of research findings for various tasks.
            Your job is to aggregate these findings into a cohesive, well-structured research report.
            The report should be in JSON format with fields: "summary", "key_findings", "details".
            Ensure the JSON is valid.""",
            llm=llm
        )

