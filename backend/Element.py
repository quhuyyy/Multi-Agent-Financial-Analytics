from typing import Any
from pydantic import BaseModel

class Element(BaseModel):
    type: str
    page_content: Any