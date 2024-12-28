from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain.output_parsers import JsonOutputToolsParser
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import json
import os

load_dotenv()

class ResearchFields(BaseModel):
    research_field: str = Field(..., description="Field of the research paper")
    research_topic: str = Field(..., description="Topic of the research paper")
    research_type: str = Field(..., description="Type of the research paper")
    authors: List[str] = Field(..., description="Authors of the research paper")
    affiliation: List[str] = Field(..., description="Affiliation of the authors")

class ResearchSections(BaseModel):
    title: str = Field(..., description="Title of the research paper")
    authors: List[str] = Field(..., description="Authors of the research paper")
    affiliation: List[str] = Field(..., description="Affiliation of the authors")
    abstract: str = Field(..., description="Abstract of the research paper")
    keywords: List[str] = Field(..., description="Keywords of the research paper")
    introduction: str = Field(..., description="Introduction of the research paper")
    related_work: str = Field(..., description="Related work of the research paper")
    research_objectives: str = Field(..., description="Research objectives of the research paper")
    literature_review: str = Field(..., description="Literature review of the research paper")
    methodology: str = Field(..., description="Methodology of the research paper")
    comparative_analysis: Optional[str] = Field(..., description="Comparative analysis of the research paper")
    future_outcomes: str = Field(..., description="Future outcomes of the research paper")
    future_objectives: str = Field(..., description="Future objectives of the research paper")
    conclusion: str = Field(..., description="Conclusion of the research paper")
    references: List
    bibliography: List
    cititations: List[str]
    

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(..., description="Search queries for the research paper")

class ResearchAgent:
    def __init__(self, research: ResearchFields):
        self.topic = research.research_topic
        self.research_type = research.research_type
        self.research_field = research.research_field
        self.authors = research.authors
        self.affiliation = research.affiliation
        
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )
    
    def _planner(self):
        prompt_template = """
        Ypu are a Planner Agent, who is assigned to create plan for each sections of the research paper.
        The Topic is {topic} and it is related to the filed of {research_field} and the research type is {research_type}.
        The Authors are {authors} and their affiliation is {affiliation}.
        Here are the sections with required format of responses required for each section: {sections}
        Note:
        - The response should be in the format of the sections provided.
        - Each section should have appropriate amount of content.
        - Each section should be unique and should not be repeated.
        """
        
        messages = [
            SystemMessage(content=prompt_template.format(
                topic=self.topic,
                research_field=self.research_field,
                research_type=self.research_type,
                authors=self.authors,
                affiliation=self.affiliation,
                sections=ResearchSections.model_json_schema()
            )),
            HumanMessage(
                content=f"Create a plan for the research paper on the topic: {self.topic}"
            )
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _search_queries(self):
        prompt_template = """
        Here is the plan for the research paper: {plan}
        Your task is to provide the search queries in strict JSON format.
        The queries should not be more than 10.
        Instructions:
        - The output must strictly adhere to the following JSON schema:
        ```json
        {{
            "search_queries": ["query1", "query2", "query3", ...]
        }}
        ```
        - Do not include any explanation, comments, or extra text. 
        - The response must be a valid JSON object and nothing else.
        [IMPORTANT]
        - The response should be in the JSON format as shown above.
        - It should follow this JSON schema: {schema}.
        """

        messages = [
            SystemMessage(content=prompt_template.format(
                plan=self._planner(),
                schema=SearchQueries.model_json_schema()
            )),
            HumanMessage(
                content=f"Create search queries for the research paper on the topic: {self.topic}"
            )
        ]
        
        # Call the LLM
        response = self.llm.invoke(messages)
        print(f"Response: {response.content}")

        # Get the response content and strip any leading/trailing spaces
        cleaned_response = response.content.strip()

        # Check if the response content starts with "```json" and ends with "```"
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()  # Remove the '```json' part

        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()  # Remove the '```' at the end

        # Try to load the cleaned content into a valid JSON object
        try:
            # If the content is already valid JSON (no extra markers), it will load directly
            response_json = json.loads(cleaned_response)
            print("JSON Loaded Successfully")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            response_json = {}

        # Return the cleaned or valid JSON response
        return response_json
    
    
    def _run(self):
        planner = self._planner()
        print(planner)
        
        queries = self._search_queries()
        print(queries)
        

if __name__ == "__main__":
    research = ResearchFields(
        research_field="Artificial Intelligence",
        research_topic="Agentic AI Vs Retrieval Augmented",
        research_type="Comparative Analysis",
        authors=["John Doe", "Jane Doe"],
        affiliation=["University of AI"]
    )
    
    agent = ResearchAgent(research)
    agent._run()