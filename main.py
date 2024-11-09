from fastapi import FastAPI
from schema import *

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/compare")
async def compare_responses(input: CompareRequest) -> CompareResponse:
    print(input.a, input.b, input.c)
    return {
        "result": "Comparing..."
    }
