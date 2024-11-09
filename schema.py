from pydantic import BaseModel

class CompareRequest(BaseModel):
    a: str
    b: str
    c: str

class CompareResponse(BaseModel):
    result: str
