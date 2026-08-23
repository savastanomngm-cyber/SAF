"""Cache AI outcomes (supply chain + pipeline) to a local folder.
Files are stored as timestamped JSON so they're human-readable and git-friendly."""
import json, re
from pathlib import Path
from datetime import datetime

OUTCOMES_DIR = Path(__file__).resolve().parent.parent.parent / "ai_outcomes"
KINDS = ("supply_chain", "pipeline")


def _slug(text, maxlen=50):
    """Make a filesystem-safe slug from a trend/ticker name."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(text)).strip("-").lower()
    return text[:maxlen] or "untitled"


def _dir(kind):
    d = OUTCOMES_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_outcome(kind, name, data):
    """Save an outcome to ai_outcomes/<kind>/<timestamp>_<slug>.json.
    Returns the filename."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{_slug(name)}.json"
    payload = {
        "kind": kind,
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "data": data,
    }
    path = _dir(kind) / fname
    path.write_text(json.dumps(payload, indent=2, default=str))
    return fname


def list_outcomes(kind):
    """List saved outcomes for a kind, newest first.
    Returns [{filename, name, saved_at, size_kb}]."""
    if kind not in KINDS:
        return []
    d = _dir(kind)
    rows = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            meta = json.loads(p.read_text())
            rows.append({
                "filename": p.name,
                "name": meta.get("name", p.stem),
                "saved_at": meta.get("saved_at", ""),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            continue
    return rows


def load_outcome(kind, filename):
    """Load a specific saved outcome. Returns the full payload or None."""
    if kind not in KINDS:
        return None
    path = _dir(kind) / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def delete_outcome(kind, filename):
    """Delete a saved outcome. Returns True if removed."""
    if kind not in KINDS:
        return False
    path = _dir(kind) / filename
    if path.exists():
        path.unlink()
        return True
    return False