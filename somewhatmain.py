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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama-based LLM response Evaluator")

class LLMResponse(BaseModel):
    llm_name: str
    prompt: str
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
            print(evaluation_text)
            
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

    def create_evaluation_prompt(self, request: EvaluationRequest) -> str:
        """Create a structured prompt optimized for Llama's instruction format"""
        prompt = f"""[INST] You are an expert at evaluating language model outputs. 
Please evaluate the following responses to this prompt: "{request.prompt}"

{self._format_responses(request.responses)}

Evaluate each response on these criteria:
1. Relevance (0-10)
2. Accuracy (0-10)
3. Coherence (0-10)
4. Completeness (0-10)

Provide your evaluation in this exact format:
SCORES:
<llm_name>:
- Relevance: <score>
- Accuracy: <score>
- Coherence: <score>
- Completeness: <score>

REASONING:
<your detailed reasoning for the scores>

FINAL RANKING:
1. <llm_name>: <average_score>
2. <llm_name>: <average_score>
... [/INST]"""
        return prompt

    def _format_responses(self, responses: List[LLMResponse]) -> str:
        formatted = "\nResponses to evaluate:\n"
        for idx, resp in enumerate(responses, 1):
            formatted += f"\n{resp.llm_name} Response:\n{resp.response}\n"
        return formatted

    def _parse_evaluation(self, eval_text: str) -> Dict:

        try:
            # Initialize default structure
            result = {
                "criteria_scores": {},
                "reasoning": "No reasoning provided",
                "rankings": [],
                "final_response" : eval_text
            }
        
            # Split into sections, handling missing sections gracefully
            sections = eval_text.split('\n\n')
        
            # Parse scores section
            scores_section = None
            for section in sections:
                if section.strip().startswith('SCORES:'):
                    scores_section = section
                    break
                
            if scores_section:
                current_llm = None
                scores = {}
            
                for line in scores_section.split('\n'):
                    line = line.strip()
                    if not line or line == 'SCORES:':
                        continue
                    
                    if ':' in line:
                        if '-' not in line:  # This is an LLM name
                            current_llm = line.replace(':', '').strip()
                            scores[current_llm] = {
                                'relevance': 0.0,
                                'accuracy': 0.0,
                                'coherence': 0.0,
                                'completeness': 0.0
                            }
                        elif current_llm:  # This is a criterion score
                            try:
                                criterion, score = [x.strip() for x in line.split(':')]
                                criterion = criterion.replace('-', '').lower()
                                # Handle any non-numeric scores gracefully
                                try:
                                    score = float(score)
                                except ValueError:
                                    score = 0.0
                                scores[current_llm][criterion] = score
                            except Exception as e:
                                logger.warning(f"Failed to parse score line '{line}': {str(e)}")
                                continue
            
                result["criteria_scores"] = scores
        
            # Parse reasoning section
            reasoning_section = None
            for section in sections:
                if section.strip().startswith('REASONING:'):
                    reasoning_section = section
                    break
                
            if reasoning_section:
                result["reasoning"] = reasoning_section.replace('REASONING:', '').strip()
        
            # Calculate rankings
            rankings = []
            for llm_name, llm_scores in result["criteria_scores"].items():
                try:
                    # Calculate average score, defaulting to 0.0 if there's an issue
                    scores_list = [score for score in llm_scores.values() if isinstance(score, (int, float))]
                    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
                    rankings.append({
                        "llm_name": llm_name,
                        "score": round(float(avg_score), 2)
                    })
                except Exception as e:
                    logger.error(f"Error calculating average for {llm_name}: {str(e)}")
                    rankings.append({
                        "llm_name": llm_name,
                        "score": 0.0
                    })
        
            # Sort rankings by score in descending order
            rankings.sort(key=lambda x: x["score"], reverse=True)
            result["rankings"] = rankings
        
            return result
        
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            # Return a valid structure even in case of error
            return {
                "criteria_scores": {},
                "reasoning": f"Error parsing evaluation: {str(e)}",
                "rankings": []
            }

llama_evaluator = LlamaEvaluator()

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_llm_responses(request: EvaluationRequest):
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