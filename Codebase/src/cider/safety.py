from __future__ import annotations

BAN = [
    "virus", "viral", "toxin", "venom", "resistance", "antibiotic", "pathogen", "infect",
    "immune", "virulence", "host-entry", "host entry", "escape", "receptor",
]

def is_safe_text(s: str) -> bool:
    t = str(s).lower()
    return not any(k in t for k in BAN)
