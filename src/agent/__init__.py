"""AI support-triage agent for AmazonHelp (Customer Support on Twitter dataset)."""
import sys as _sys

# Tweets contain arbitrary unicode (emoji, non-English scripts even after
# filtering, curly quotes). Windows consoles default to cp1252 and crash on
# print() for those bytes; reconfigure defensively so scripts/*.py never die
# on a stray character while logging progress.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
