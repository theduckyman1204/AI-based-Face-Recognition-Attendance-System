from pydantic import BaseModel


class PersonEnrollResponse(BaseModel):
    id: int
    name: str
    message: str | None = None
    samples: int | None = None


class PersonEnrollRequest(BaseModel):
    name: str
