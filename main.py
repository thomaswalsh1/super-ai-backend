from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
import uvicorn
import logging
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import platform
import os
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama-based LLM response Evaluator")

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

class EvaluationResult(BaseModel):
    rankings: List[Ranking]
    reasoning: str
    evaluation_time: float
    criteria_scores: Dict[str, Dict[str, float]]

class LlamaEvaluator:
    def __init__(self):
        """Initialize Llama model and tokenizer with MacBook optimization"""
        self.setup_device()
        
        try:
            self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Set up padding token
            logger.info("Setting up padding token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model from {self.model_name}")
            if self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=None
                )
                self.model.to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model.to(self.device)
            
            # Ensure model knows about padding token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            logger.info(f"Successfully loaded Llama model on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def setup_device(self):
        """Setup the appropriate device for MacBook"""
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            logger.info("Using CPU - No MPS detected")
        
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


    @torch.no_grad()
    async def evaluate_responses(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate and rank the LLM responses using Llama"""
        start_time = datetime.now()
        
        try:
            # Create evaluation prompt
            evaluation_prompt = self.create_evaluation_prompt(request)
            
            # Tokenize input
            encoded_input = self.tokenizer(
                evaluation_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move input tensors to device
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Generate response
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Move output to CPU for decoding
            outputs = outputs.cpu()
            
            # Decode the generated text
            evaluation_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract the response part after the instruction
            evaluation_text = evaluation_text.split('[/INST]')[-1].strip()
            
            if self.device == "mps":
                torch.mps.empty_cache()
            
            parsed_evaluation = self._parse_evaluation(evaluation_text)
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Create properly structured EvaluationResult
            rankings = [Ranking(llm_name=r["llm_name"], score=r["score"]) 
                       for r in parsed_evaluation["rankings"]]
            
            return EvaluationResult(
                rankings=rankings,
                reasoning=parsed_evaluation["reasoning"],
                evaluation_time=evaluation_time,
                criteria_scores=parsed_evaluation["criteria_scores"]
            )
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}")
            raise HTTPException(status_code=500, detail=str(e))


    @torch.no_grad()
    def _parse_evaluation(self, eval_text: str) -> Dict:
        """
        Parse the evaluation text from LLM response into structured format.
        """
        try:
            # Initialize result structure
            result = {
                "criteria_scores": {},
                "reasoning": "",
                "rankings": []
            }
        
            # Split by LLM responses using "**" as delimiter
            llm_sections = eval_text.split("**")
        
            # Parse scores and reasoning for each LLM
            for section in llm_sections:
                if "SCORES:" in section and "Response" in section:
                    # Extract LLM name
                    llm_name = section.split("Response")[0].strip()
                
                    # Initialize scores dict for this LLM
                    scores = {}
                
                    # Find scores section
                    scores_text = section.split("SCORES:")[1].split("REASONING:")[0]
                
                    # Parse individual scores
                    for line in scores_text.split("\n"):
                        if ":" in line and "-" in line:
                            criterion = line.split(":")[0].replace("-", "").strip().lower()
                            try:
                                score = float(line.split(":")[1].strip())
                                scores[criterion] = score
                            except ValueError:
                                scores[criterion] = 0.0
                
                    result["criteria_scores"][llm_name] = scores
                
                    # Extract reasoning if present
                    if "REASONING:" in section:
                        reasoning = section.split("REASONING:")[1].split("SCORES:", 1)[0].strip()
                        if reasoning:
                            if result["reasoning"]:
                                result["reasoning"] += "\n\n"
                            result["reasoning"] += f"{llm_name} Reasoning: {reasoning}"
        
            # Parse final ranking section
            if "FINAL RANKING:" in eval_text:
                ranking_section = eval_text.split("FINAL RANKING:")[1].split("Note")[0]
                for line in ranking_section.split("\n"):
                    if ":" in line and "." in line:
                        try:
                            llm_name = line.split(".")[1].split(":")[0].strip()
                            score = float(line.split(":")[1].strip())
                            result["rankings"].append({
                                "llm_name": llm_name,
                                "score": score
                            })
                        except (ValueError, IndexError):
                            continue
        
            # If no rankings were found, calculate them from scores
            if not result["rankings"]:
                for llm_name, scores in result["criteria_scores"].items():
                    if scores:
                        avg_score = sum(scores.values()) / len(scores)
                        result["rankings"].append({
                            "llm_name": llm_name,
                            "score": round(float(avg_score), 2)
                        })
                # Sort rankings by score in descending order
                result["rankings"].sort(key=lambda x: x["score"], reverse=True)
        
            return result
        
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            return {
                "criteria_scores": {},
                "reasoning": f"Error parsing evaluation: {str(e)}",
                "rankings": []
            }

    def create_evaluation_prompt(self, request: EvaluationRequest) -> str:
        """Create a strictly formatted prompt that enforces structure"""
    
        # Format responses with clear numbering
        responses_text = ""
        for idx, resp in enumerate(request.responses, 1):
            responses_text += f"\nResponse {idx}. {resp.llm_name}:\n{resp.response}\n"
    
        prompt = f"""[INST] Evaluate these {len(request.responses)} language model responses to the prompt: "{request.prompt}"

    {responses_text}

    You MUST follow this EXACT format in your response:

    EVALUATION FOR {request.responses[0].llm_name}:
    SCORES:
    - Relevance: (0-10)
    - Accuracy: (0-10)
    - Coherence: (0-10)
    - Completeness: (0-10)
    REASONING: (2-3 sentences explaining these scores)

    EVALUATION FOR {request.responses[1].llm_name}:
    SCORES:
    - Relevance: (0-10)
    - Accuracy: (0-10)
    - Coherence: (0-10)
    - Completeness: (0-10)
    REASONING: (2-3 sentences explaining these scores)

    [Continue same format for each response]

    FINAL RANKING:
    1. [Model Name]: [Average Score]
    2. [Model Name]: [Average Score]
    3. [Model Name]: [Average Score]
    4. [Model Name]: [Average Score]
    5. [Model Name]: [Average Score]

    Rules:
    1. You must evaluate EVERY response
    2. All scores must be numbers between 0 and 10
    3. You must provide reasoning for each response
    4. You must include a final ranking of ALL responses
    5. Use EXACT headings: "EVALUATION FOR", "SCORES:", "REASONING:", "FINAL RANKING:"
    6. Calculate average scores as (Relevance + Accuracy + Coherence + Completeness) / 4

    Begin your evaluation now, following the exact format above. [/INST]"""
        return prompt

    def _format_responses(self, responses: List[LLMResponse]) -> str:
        formatted = "\nResponses to evaluate:\n"
        for idx, resp in enumerate(responses, 1):
            formatted += f"\n{resp.llm_name} Response:\n{resp.response}\n"
        return formatted

    def _parse_evaluation(self, eval_text: str) -> Dict:
        """Parse evaluation text with strict format checking"""
        try:
            result = {
                "criteria_scores": {},
                "reasoning": "",
                "rankings": []
            }
            
            # Split into individual evaluations
            evaluations = eval_text.split("EVALUATION FOR")
            all_reasoning = []
            
            # Process each evaluation section
            for eval_section in evaluations[1:]:  # Skip first empty split
                try:
                    # Extract LLM name
                    llm_name = eval_section.split(":")[0].strip()
                    
                    # Extract scores
                    if "SCORES:" in eval_section:
                        scores_section = eval_section.split("SCORES:")[1].split("REASONING:")[0]
                        scores = {}
                        
                        for line in scores_section.split("\n"):
                            if "-" in line and ":" in line:
                                criterion = line.split(":")[0].replace("-", "").strip().lower()
                                try:
                                    score = float(line.split(":")[1].strip())
                                    scores[criterion] = score
                                except ValueError:
                                    scores[criterion] = 0.0
                        
                        result["criteria_scores"][llm_name] = scores
                    
                    # Extract reasoning
                    if "REASONING:" in eval_section:
                        reasoning = eval_section.split("REASONING:")[1].split("EVALUATION FOR")[0]
                        reasoning = reasoning.split("FINAL RANKING:")[0].strip()
                        all_reasoning.append(f"{llm_name}: {reasoning}")
                
                except Exception as e:
                    logger.error(f"Error processing evaluation section: {str(e)}")
                    continue
            
            # Combine all reasoning
            result["reasoning"] = "\n\n".join(all_reasoning)
            
            # Extract final ranking
            if "FINAL RANKING:" in eval_text:
                ranking_section = eval_text.split("FINAL RANKING:")[1].strip()
                for line in ranking_section.split("\n"):
                    if ":" in line and any(char.isdigit() for char in line):
                        try:
                            # Remove number prefix and split by colon
                            line = re.sub(r'^\d+\.\s*', '', line)
                            llm_name, score = line.split(":")
                            result["rankings"].append({
                                "llm_name": llm_name.strip(),
                                "score": float(score.strip())
                            })
                        except (ValueError, IndexError) as e:
                            logger.error(f"Error parsing ranking line '{line}': {str(e)}")
                            continue
            
            # If no rankings found, calculate from scores
            if not result["rankings"] and result["criteria_scores"]:
                for llm_name, scores in result["criteria_scores"].items():
                    if scores:
                        avg_score = sum(scores.values()) / len(scores)
                        result["rankings"].append({
                            "llm_name": llm_name,
                            "score": round(float(avg_score), 2)
                        })
                # Sort by score descending
                result["rankings"].sort(key=lambda x: x["score"], reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            raise ValueError(f"Failed to parse evaluation output: {str(e)}")

llama_evaluator = LlamaEvaluator()

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_responses(request: EvaluationRequest):
    """Endpoint to evaluate and rank LLM responses"""
    logger.info(f"Received evaluation request for prompt: {request.prompt[:100]}...")
    return await llama_evaluator.evaluate_responses(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_info = ""
    if llama_evaluator.device == "mps":
        memory_info = f", MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB"
    
    return {
        "status": "healthy",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "device": llama_evaluator.device,
        "system": platform.platform(),
        "memory_info": memory_info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)