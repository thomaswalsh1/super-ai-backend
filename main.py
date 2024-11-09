from fastapi import FastAPI
from schema import *



app = FastAPI()
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/compare")
async def compare_responses(input: CompareRequest) -> CompareResponse:
    print(input.a, input.b, input.c)
    return {
        "result": "Comparing..."
    }
