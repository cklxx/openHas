from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Claims:
    sub: str
    exp: int
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthError:
    code: Literal['EXPIRED', 'INVALID_SIG', 'MALFORMED']
    detail: str = ''
