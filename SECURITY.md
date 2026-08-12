# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose local video
content, API credentials, or arbitrary files.

Use GitHub's private vulnerability reporting for this repository. Include the
affected version, reproduction steps, impact, and any suggested mitigation.

## Data boundary

FrameSeek processes videos locally by default. A cloud backend sends selected
frames and the question to the configured API endpoint. It never uploads the
entire source video unless a custom backend explicitly implements that behavior.
FrameSeek refuses to send an API key over plain HTTP to a non-loopback endpoint;
use HTTPS for remote model servers.

Index files can contain captions derived from private videos. New indexes store
only the source video's filename, but older indexes may contain its full path.
Treat indexes as sensitive data and do not commit them to a public repository.

Before a frame is sent to any backend, FrameSeek requires its path to stay
inside the index directory and verifies its SHA-256 digest when the index
provides one. Keep `index.json` and its `frames/` directory together; modifying
an indexed frame invalidates that index by design.

FFmpeg and ffprobe subprocesses have a bounded runtime so malformed media cannot
stall indexing indefinitely. SDK users can adjust the limit with
`FFmpegMediaTool(timeout_seconds=...)` for unusually slow local workloads.
