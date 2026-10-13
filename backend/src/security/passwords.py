"""
src/security/passwords.py

Password hashing with scrypt from the standard library (memory-hard, no extra
dependency). Format: scrypt$N$r$p$<salt_b64>$<hash_b64>, so parameters can be
raised later and old hashes still verify.
"""
import base64
import hashlib
import hmac
import os

_N, _R, _P = 2 ** 14, 8, 1
MIN_PASSWORD_LENGTH = 10


class WeakPassword(ValueError):
    pass


def validate_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPassword(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.lower() in {"password123", "1234567890", "qwertyuiop", "adminadmin"}:
        raise WeakPassword("Password is too common.")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode("ascii")
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r),
                                p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# Verified against when the username doesn't exist, so "no such user" and "wrong
# password" take the same time (no username enumeration via timing).
DUMMY_HASH = hash_password("dummy-password-for-timing-equalisation")
