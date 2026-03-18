"""Authentication — pure functions, Result-based error handling."""

import base64
import hashlib
import hmac
import json
from typing import cast

from src.domain_types.auth import AuthError, Claims
from src.domain_types.result import Result

_JWT_PARTS = 3


def _decode_payload(token: str) -> dict[str, object] | None:
    parts = token.split('.')
    if len(parts) != _JWT_PARTS:
        return None
    try:
        payload_b64 = parts[1] + '=' * (-len(parts[1]) % 4)
        raw = base64.urlsafe_b64decode(payload_b64)
        return cast(dict[str, object], json.loads(raw))
    except (ValueError, json.JSONDecodeError):
        return None


def _verify_sig(token: str, secret: str) -> bool:
    parts = token.split('.')
    if len(parts) != _JWT_PARTS:
        return False
    signing_input = f"{parts[0]}.{parts[1]}"
    expected = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    sig = base64.urlsafe_b64decode(parts[2] + '=' * (-len(parts[2]) % 4))
    return hmac.compare_digest(expected, sig)


def _validate_payload(
    payload: dict[str, object], now: float
) -> Result[Claims, AuthError]:
    exp = payload.get('exp')
    if not isinstance(exp, int | float):
        return ('err', AuthError(code='MALFORMED', detail='missing exp'))
    if float(exp) < now:
        return ('err', AuthError(code='EXPIRED', detail=str(exp)))
    return _extract_claims(payload, int(exp))


def _extract_claims(
    payload: dict[str, object], exp: int
) -> Result[Claims, AuthError]:
    sub = payload.get('sub')
    if not isinstance(sub, str):
        return ('err', AuthError(code='MALFORMED', detail='missing sub'))
    raw_roles = payload.get('roles', ())
    if not isinstance(raw_roles, list) or not all(
        isinstance(r, str) for r in cast(list[object], raw_roles)
    ):
        return ('err', AuthError(code='MALFORMED', detail='bad roles'))
    return ('ok', Claims(sub=sub, exp=exp, roles=tuple(cast(list[str], raw_roles))))


def authenticate(token: str, secret: str, now: float) -> Result[Claims, AuthError]:
    payload = _decode_payload(token)
    if payload is None:
        return ('err', AuthError(code='MALFORMED'))
    if not _verify_sig(token, secret):
        return ('err', AuthError(code='INVALID_SIG'))
    return _validate_payload(payload, now)
