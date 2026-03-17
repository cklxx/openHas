"""Property-based tests for authentication."""

import base64
import hashlib
import hmac
import json

from hypothesis import given
from hypothesis import strategies as st
from src.core.auth import authenticate


def _encode_token(payload: dict[str, object], secret: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b'=').decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()
    signing_input = f"{header}.{body}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b'=').decode()
    return f"{header}.{body}.{sig_b64}"


_valid_claims = st.fixed_dictionaries({
    'sub': st.text(min_size=1, max_size=50),
    'exp': st.integers(min_value=1_000_000_000, max_value=9_999_999_999),
    'roles': st.lists(st.text(min_size=1, max_size=20), max_size=5).map(list),
})


@given(st.binary(min_size=1, max_size=200))
def test_authenticate_rejects_garbage(data: bytes) -> None:
    result = authenticate(data.decode('latin-1'), secret='test', now=0.0)
    assert result[0] == 'err'


@given(_valid_claims)
def test_roundtrip(payload: dict[str, object]) -> None:
    token = _encode_token(payload, 'secret')
    result = authenticate(token, secret='secret', now=0.0)
    assert result[0] == 'ok'


@given(_valid_claims)
def test_expired_token(payload: dict[str, object]) -> None:
    token = _encode_token(payload, 'secret')
    far_future = 99_999_999_999.0
    result = authenticate(token, secret='secret', now=far_future)
    assert result[0] == 'err' and result[1].code == 'EXPIRED'


@given(_valid_claims)
def test_wrong_secret_rejected(payload: dict[str, object]) -> None:
    token = _encode_token(payload, 'right-secret')
    result = authenticate(token, secret='wrong-secret', now=0.0)
    assert result[0] == 'err' and result[1].code == 'INVALID_SIG'
