from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional

class JokeOutputSchema(BaseModel):
    pass