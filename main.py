from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/compare")
async def compare_responses():
    return {"message": "Comparing..."}
