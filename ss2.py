import requests
from pydantic import BaseModel, field_validator
from openai import OpenAI
from typing import Optional, Union, Dict

# from datetime import datetime
import json


class PaperAnalysis(BaseModel):
    title: str
    publication_date: Optional[str]
    is_relevant: bool
    relevance_explanation: str
    study_type: Optional[str]
    sample_size: Optional[int]
    key_findings: Optional[str]
    stone_free_rate: Optional[Union[float, Dict[str, float]]]
    methodology_quality: Optional[str]
    limitations: Optional[str]
    pmid: Optional[str] = None
    abstract: Optional[str] = None

    @field_validator("limitations")
    def convert_list_to_string(cls, v):
        if isinstance(v, list):
            return ". ".join(v)
        return v

    @field_validator("key_findings")
    def convert_dict_to_string(cls, v):
        if isinstance(v, dict):
            return ". ".join(f"{k}: {v}" for k, v in v.items())
        return v


class PaperEvaluator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.analyzed_papers = []

    def evaluate_paper(
        self,
        title: str,
        abstract: str,
        publication_date: str = None,
        research_question: str = None,
    ) -> PaperAnalysis:
        # Use default research question if none provided
        if not research_question:
            research_question = "Does displacement of lower pole stones during retrograde intrarenal surgery improve stone-free status?"

        prompt = f"""
        Research Question: {research_question}
        
        Primary Outcome: Effectiveness of translocating lower pole renal stones to other locations (upper pole or interpolar region) during RIRS
        
        Secondary Outcomes: Operative time, laser energy usage, and complications

        Please analyze the following paper for inclusion in a systematic review and meta-analysis, focusing on the above outcomes:
        
        Paper Title: {title}
        Publication Date: {publication_date}
        Abstract: {abstract}

        Provide a detailed analysis in JSON format with these exact fields:
        - title: the paper title
        - publication_date: the publication date
        - is_relevant: boolean indicating if the paper is relevant to the research question
        - relevance_explanation: explanation of relevance
        - study_type: type of study (RCT, cohort, etc.)
        - sample_size: number of patients (null if not mentioned)
        - stone_free_rate: can be either a single decimal number or a dictionary of group rates (e.g., {{"in_situ": 0.85, "relocation": 0.91}})
        - methodology_quality: assessment of methodology
        - limitations: study limitations as a single string or list of limitations
        - key_findings: main findings related to the research question

        Extract and analyze the paper information systematically, focusing on stone-free rates and methodology.
        """

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical research analyst specializing in urology and meta-analysis. Extract and analyze paper information systematically.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=PaperAnalysis,
        )

        # Get the parsed PaperAnalysis object directly
        analysis = completion.choices[0].message.parsed
        self.analyzed_papers.append(analysis)
        return analysis

    def save_analysis(self, filename="meta_analysis_data.json"):
        """Save all analyzed papers to a JSON file in root"""
        with open(filename, "w") as f:
            json.dump(
                [paper.model_dump() for paper in self.analyzed_papers], f, indent=2
            )
