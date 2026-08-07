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

Index files can contain captions and paths derived from private videos. Treat
them as sensitive data and do not commit them to a public repository.
