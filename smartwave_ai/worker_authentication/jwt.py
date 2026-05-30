from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from smartwave_ai.worker_authentication.config import JWT_ISSUER, JWT_SECRET
from smartwave_ai.worker_authentication.models import AuthClaims, UserRole


class JwtAuthError(Exception):
    error_code = "ERR_AUTH_INVALID"


class JwtExpiredError(JwtAuthError):
    error_code = "ERR_AUTH_EXPIRED"


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def jwt_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_hs256_jwt(token: str, secret: str = JWT_SECRET) -> AuthClaims:
    parts = token.split(".")
    if len(parts) != 3:
        raise JwtAuthError("JWT must have header, payload, and signature.")
    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise JwtAuthError("JWT payload is malformed.") from exc

    if header.get("alg") != "HS256":
        raise JwtAuthError("Only HS256 is supported in local development.")
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        f"{header_b64}.{payload_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    supplied_signature = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise JwtAuthError("JWT signature is invalid.")

    exp = payload.get("exp")
    if exp is not None and int(exp) < int(time.time()):
        raise JwtExpiredError("JWT is expired.")

    subject = str(payload.get("sub") or payload.get("worker_id") or "")
    role = payload.get("role")
    if not subject or role not in {item.value for item in UserRole}:
        raise JwtAuthError("JWT missing required subject or role.")

    return AuthClaims(
        subject=subject,
        role=role,
        worker_id=payload.get("worker_id") or subject,
        exp=exp,
        issuer=payload.get("iss"),
        jwt_fingerprint=jwt_fingerprint(token),
    )


def create_dev_jwt(payload: dict[str, Any], secret: str = JWT_SECRET) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": JWT_ISSUER, **payload}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{header_b64}.{payload_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"

