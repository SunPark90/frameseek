class FrameSeekError(Exception):
    """Base exception for expected, user-actionable failures."""


class DependencyError(FrameSeekError):
    """Raised when an optional executable or Python package is unavailable."""


class MediaError(FrameSeekError):
    """Raised when media probing or frame extraction fails."""


class IndexFormatError(FrameSeekError):
    """Raised when an index is missing required or valid fields."""


class BackendError(FrameSeekError):
    """Raised when a model backend cannot complete a request."""


class BackendProtocolError(BackendError):
    """Raised when a backend returns an unverifiable response."""


class EvidenceError(FrameSeekError):
    """Raised when answer evidence does not match inspected frames."""
