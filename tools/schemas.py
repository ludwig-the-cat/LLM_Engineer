from pydantic import BaseModel, Field
from typing import Optional

class JokeOutputSchema(BaseModel):
    """
    Joke to tell user
    """
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description='The punchline of the joke')
    rating: Optional[int] = Field(description='Rate from 1 to 10')
