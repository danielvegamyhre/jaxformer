
from typing import Protocol

class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]:
        pass

def preprocess(filepath: str, tokenizer: TokenizerInterface) -> list[int]:
    with open(filepath, 'r') as fp:
        return tokenizer.encode(fp.read())