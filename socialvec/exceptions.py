"""Exceptions raised by SocialVec."""


class SocialVecError(Exception):
    """Base class for all SocialVec errors."""


class InvalidUserIdError(SocialVecError, ValueError):
    """The value given is not a valid Twitter user ID or username."""


class UserNotFoundError(SocialVecError, KeyError):
    """The user is not in the loaded SocialVec model (or its metadata)."""

    def __str__(self):
        # KeyError wraps its message in quotes; show it like a normal error.
        return str(self.args[0]) if self.args else ""
