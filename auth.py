import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from jose import JWTError, jwt


load_dotenv(Path(__file__).resolve().parent / ".env")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if len(SECRET_KEY) < 32 or SECRET_KEY.startswith("replace-with-"):
    raise RuntimeError("Set JWT_SECRET_KEY to a random secret of at least 32 characters")

ALGORITHM = "HS256"

TOKEN_EXPIRE_MINUTES = 60
USERS = {
    "alice": {
        "password_hash": "pbkdf2_sha256$600000$c59f11c02d0feb9cfe7bbe7c4e012b46$221b0d6f7ad31125354979c3dcd7c144df86e9b3f4f7dfc8614209320d90ff04",
        "tenant_id": "acme",
        "role": "employee",
        "active": True,
        "token_version": 0,
    },

    "bob": {
        "password_hash": "pbkdf2_sha256$600000$a2f4c266811c022b4c9e173234f196b2$4ba8698459c9361774ccb9d9a901e2e3e03418e0b5ef5851bdbb4472fa2751a7",
        "tenant_id": "acme",
        "role": "hr",
        "active": True,
        "token_version": 0,
    },

    "carol": {
        "password_hash": "pbkdf2_sha256$600000$6b82f1ea25fe42886879fc641f639e53$259dd14000150eeecb65211810c5a1dad1b8fdc365f71b01d065c47a73a15adb",
        "tenant_id": "globex",
        "role": "hr",
        "active": True,
        "token_version": 0,
    },
}


def verify_password(password, stored_hash):
    algorithm, iterations, salt, expected = stored_hash.split("$")
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
    )
    return hmac.compare_digest(actual.hex(), expected)


def authenticate_user(username, password):
    user = USERS.get(username)
    # Do the hash work even for unknown users to reduce username timing leaks.
    stored_hash = user["password_hash"] if user else USERS["alice"]["password_hash"]
    password_valid = verify_password(password, stored_hash)
    if not user or not user["active"] or not password_valid:
        return None

    return {
        "user_id": username,
        "tenant_id": user["tenant_id"],
        "role": user["role"],
    }

def create_access_token(user_id, tenant_id, role):
    user = USERS.get(user_id)
    if (
        not user or not user["active"]
        or user["tenant_id"] != tenant_id or user["role"] != role
    ):
        raise ValueError("Cannot issue a token for an invalid identity")
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "exp": expire,
        "token_version": user["token_version"],
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return token


def verify_access_token(token):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require_exp": True},
        )
        for claim in ("user_id", "tenant_id", "role"):
            if not isinstance(payload.get(claim), str) or not payload[claim].strip():
                return None
        if type(payload.get("exp")) is not int:
            return None
        if payload["exp"] <= datetime.now(timezone.utc).timestamp():
            return None
        if type(payload.get("token_version")) is not int:
            return None
        # Recheck current server-side identity on every request. Disabling a user,
        # changing tenant/role, or incrementing token_version invalidates old JWTs.
        user = USERS.get(payload["user_id"])
        if (
            not user or not user["active"]
            or payload["tenant_id"] != user["tenant_id"]
            or payload["role"] != user["role"]
            or payload["token_version"] != user["token_version"]
        ):
            return None
        return payload

    except (JWTError, TypeError, ValueError, OverflowError):
        return None


if __name__ == "__main__":
    token = create_access_token(
        user_id="bob",
        tenant_id="acme",
        role="hr",
    )

    payload = verify_access_token(token)

    print("\nVerified payload:")
    print(payload)
