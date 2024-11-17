import logging
import os
from datetime import datetime
from typing import Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

origins = ["*"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama-based LLM response Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)


class LLMResponse(BaseModel):
    llm_name: str
    response: str
    metadata: Dict = {}


class EvaluationRequest(BaseModel):
    prompt: str
    responses: List[LLMResponse]


class Ranking(BaseModel):
    llm_name: str
    score: float
    scores: Dict[str, float]
    reasoning: str


class EvaluationResult(BaseModel):
    rankings: List[Ranking]
    evaluation_time: float


class LlamaEvaluator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.model_name = "meta/llama-3.1-405b-instruct"

    async def evaluate_responses(self, request: EvaluationRequest) -> EvaluationResult:
        start_time = datetime.now()
        try:
            evaluation_prompt = self.create_evaluation_prompt(request)
            response = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"],
            )
            evaluation_text = response.choices[0].message.content
            print(evaluation_text)
            evaluation_text = evaluation_text.split("[/INST]")[-1].strip()
            parsed_evaluation = self._parse_evaluation(evaluation_text)
            evaluation_time = (datetime.now() - start_time).total_seconds()
            return EvaluationResult(
                rankings=parsed_evaluation, evaluation_time=evaluation_time
            )
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _parse_evaluation(self, eval_text: str) -> List[Ranking]:
        logger.info(f"LLM Response: {eval_text}")
        rankings = []
        sections = eval_text.split("EVALUATION FOR")
        logger.info(f"Sections: {sections}")
        for section in sections[1:]:
            logger.info(f"Section: {section}")
            llm_name = section.split(":")[0].strip()
            logger.info(f"LLM Name: {llm_name}")
            scores_text = section.split("SCORES:")[1].split("REASONING:")[0]
            logger.info(f"Scores Text: {scores_text}")
            scores = {}
            for line in scores_text.split("\n"):
                if ":" in line and "-" in line:
                    criterion = line.split(":")[0].replace("-", "").strip().lower()
                    score = float(line.split(":")[1].strip())
                    scores[criterion] = score
            logger.info(f"Scores: {scores}")
            reasoning = section.split("REASONING:")[1].strip()
            logger.info(f"Reasoning: {reasoning}")
            rankings.append(
                Ranking(
                    llm_name=llm_name,
                    score=sum(scores.values()) / len(scores),
                    scores=scores,
                    reasoning=reasoning,
                )
            )
        logger.info(f"Rankings: {rankings}")
        return rankings

    def create_evaluation_prompt(self, request: EvaluationRequest) -> str:
        responses_text = ""
        for idx, resp in enumerate(request.responses, 1):
            responses_text += f"\nResponse {idx}. {resp.llm_name}:\n{resp.response}\n"
        prompt = f"""[INST] Evaluate these {len(request.responses)} language model responses to the prompt: "{request.prompt}"

    {responses_text}

    You MUST follow this EXACT format in your response:

    EVALUATION FOR {request.responses[0].llm_name}:
    SCORES:
    - Relevance: (a numerical score between 0 and 10)
    - Accuracy: (a numerical score between 0 and 10)
    - Coherence: (a numerical score between 0 and 10)
    - Completeness: (a numerical score between 0 and 10)
    REASONING: (2-3 sentences explaining these scores)

    EVALUATION FOR {request.responses[1].llm_name}:
    SCORES:
    - Relevance: (a numerical score between 0 and 10)
    - Accuracy: (a numerical score between 0 and 10)
    - Coherence: (a numerical score between 0 and 10)
    - Completeness: (a numerical score between 0 and 10)
    REASONING: (2-3 sentences explaining these scores)

    [Continue same format for each response]

    Note: You must evaluate EVERY response. All scores must be numerical values between 0 and 10. You must provide reasoning for each response. Do not use 'N/A' or any other non-numerical value as a score.

    Begin your evaluation now, following the exact format above. [/INST]"""
        return prompt


llama_evaluator = LlamaEvaluator()


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_responses(request: EvaluationRequest):
    logger.info(f"Received evaluation request for prompt: {request.prompt[:100]}...")
    return await llama_evaluator.evaluate_responses(request)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "nvidia/llama",
        "openai_api_key": "set" if os.environ.get("OPENAI_API_KEY") else "not set",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
