from typing import Literal, TypeVar

T = TypeVar('T')
E = TypeVar('E')

# Python 3.11 compat — PEP 695 `type` statement requires 3.12+
Result = tuple[Literal['ok'], T] | tuple[Literal['err'], E]
