from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
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

class EvaluationResult(BaseModel):
    rankings: List[Dict[str, float]]
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
            
            # Fixed tokenization process
            encoded_input = self.tokenizer(
                evaluation_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Adjust based on your needs
            )
            
            # Move input tensors to device
            attention_mask = encoded_input['attention_mask'].to(self.device)
            input_ids = encoded_input['input_ids'].to(self.device)
            
            
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Generate with proper input format
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
            
            return EvaluationResult(
                rankings=parsed_evaluation["rankings"],
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
            # Split the evaluation into sections
            sections = eval_text.split('\n\n')
            
            # Parse scores
            scores_section = next(s for s in sections if s.startswith('SCORES:'))
            scores = {}
            current_llm = None
            
            for line in scores_section.split('\n'):
                if ':' in line:
                    if 'SCORES:' not in line:
                        if '-' not in line:
                            current_llm = line.replace(':', '').strip()
                            scores[current_llm] = {}
                        else:
                            criterion, score = line.split(':')
                            criterion = criterion.replace('-', '').strip()
                            score = float(score.strip())
                            scores[current_llm][criterion.lower()] = score

            # Extract reasoning
            reasoning_section = next(s for s in sections if s.startswith('REASONING:'))
            reasoning = reasoning_section.replace('REASONING:', '').strip()

            # Calculate rankings
            rankings = []
            for llm_name, llm_scores in scores.items():
                avg_score = sum(llm_scores.values()) / len(llm_scores)
                rankings.append({"llm_name": llm_name, "score": avg_score})
            
            rankings.sort(key=lambda x: x["score"], reverse=True)

            return {
                "criteria_scores": scores,
                "reasoning": reasoning,
                "rankings": rankings
            }
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            raise ValueError(f"Failed to parse evaluation output: {str(e)}")

    @torch.no_grad()
    async def evaluate_responses(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate and rank the LLM responses using Llama"""
        start_time = datetime.now()
        
        try:
            evaluation_prompt = self.create_evaluation_prompt(request)
            
            # Modified tokenization and generation process
            inputs = self.tokenizer(evaluation_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.device == "mps":
                torch.mps.empty_cache()
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Move output to CPU for decoding
            outputs = outputs.cpu()
            
            evaluation_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            evaluation_text = evaluation_text.split('[/INST]')[-1].strip()
            
            if self.device == "mps":
                torch.mps.empty_cache()
            
            parsed_evaluation = self._parse_evaluation(evaluation_text)
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            return EvaluationResult(
                rankings=parsed_evaluation["rankings"],
                reasoning=parsed_evaluation["reasoning"],
                evaluation_time=evaluation_time,
                criteria_scores=parsed_evaluation["criteria_scores"]
            )
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

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