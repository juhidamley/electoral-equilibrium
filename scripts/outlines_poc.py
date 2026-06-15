"""Proof-of-concept: outlines constrained JSON generation with gpt2."""

import outlines
from pydantic import BaseModel


class Output(BaseModel):
    x: int


model = outlines.models.transformers("gpt2")
generator = outlines.generate.json(model, Output)
result = generator("Give me a JSON object")
print(result)
assert isinstance(result.x, int), f"expected int, got {type(result.x)}"
