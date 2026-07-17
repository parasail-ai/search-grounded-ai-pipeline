"""
models.py — Parasail model registry with pricing.

Pricing source: parasail.io/pricing (per 1M tokens, serverless tier).
API slugs: confirmed via docs or GET https://api.parasail.io/v1/models.

Named serverless models use Parasail-specific slugs (e.g. "gpt-oss-20b").
Open-weight models use HuggingFace IDs (e.g. "Qwen/Qwen3.6-35B-A3B").
Verify current slugs by running: python3 models.py list

Usage:
    from models import MODELS, get_model, default_model

    cfg = get_model("qwen3.6")
    agent = ParasailAgent(model=cfg["model_id"])
"""

# ─── Model registry ──────────────────────────────────────────────────────────
# Keys are short display names used in the UI / CLI.
# model_id: the string to pass to the API as "model"
# Pricing: per 1M tokens (serverless tier), USD.
# slug_confirmed: True = verified via API or official docs; False = inferred from HF ID, verify via /v1/models

PARASAIL_BASE_URL = "https://api.parasail.io/v1"

MODELS = {
    # ── GPT-OSS (OpenAI open-weight, hosted by Parasail) ─────────────────────
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b",      # slug from official Parasail quickstart snippet
        "display_name": "GPT-OSS 20B",
        "provider": "OpenAI (open-weight)",
        "pricing": {"input": 0.04, "output": 0.20, "cache_read": 0.02},
        "slug_confirmed": True,
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",     # slug from official docs (GPT-OSS 20b and 120b page)
        "display_name": "GPT-OSS 120B",
        "provider": "OpenAI (open-weight)",
        "pricing": {"input": 0.10, "output": 0.75, "cache_read": 0.055},
        "slug_confirmed": True,
    },

    # ── Qwen (Alibaba) ────────────────────────────────────────────────────────
    "qwen3.6": {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "display_name": "Qwen3.6 35B-A3B",
        "provider": "Alibaba",
        "pricing": {"input": 0.15, "output": 1.00, "cache_read": 0.05},
        "slug_confirmed": True,
    },

    # ── GLM (Zhipu AI / ZAI) ─────────────────────────────────────────────────
    "glm-5": {
        "model_id": "zai-org/GLM-5",
        "display_name": "GLM-5",
        "provider": "Zhipu AI",
        "pricing": {"input": 1.00, "output": 3.20, "cache_read": 0.20},
        "slug_confirmed": True,
    },

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    "deepseek-v4-pro": {
        "model_id": "deepseek-ai/DeepSeek-V4-Pro",
        "display_name": "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "pricing": {"input": 1.74, "output": 3.48, "cache_read": 0.10},
        "slug_confirmed": True,
    },
}

# The default model used when none is specified
DEFAULT_MODEL = "gpt-oss-20b"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_model(key: str) -> dict:
    """Return model config by short key. Raises KeyError with hint on miss."""
    if key in MODELS:
        return MODELS[key]
    raise KeyError(f"Unknown model key: {key!r}. Available: {list(MODELS)}")


def default_model() -> dict:
    return MODELS[DEFAULT_MODEL]


def model_id(key: str) -> str:
    """Shorthand: return just the model_id string for a given key."""
    return get_model(key)["model_id"]


def list_models() -> list[dict]:
    """Return all models as a list, sorted by input price ascending."""
    return sorted(
        [{"key": k, **v} for k, v in MODELS.items()],
        key=lambda m: m["pricing"]["input"],
    )


def list_models_from_api(api_key: str) -> list:
    """
    Fetch the live model list directly from Parasail's OpenAI-compatible endpoint.
    Parasail exposes GET /v1/models — same interface as the OpenAI SDK's client.models.list().
    Returns a list of model IDs available on your API key's tier.
    """
    from openai import OpenAI
    # Point the standard OpenAI client at Parasail's base URL — no other changes needed
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.parasail.io/v1",
    )
    return [m.id for m in client.models.list().data]


# ─── CLI: list models + optionally probe /v1/models for live slugs ───────────

if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        print(f"\n{'Key':<18} {'Model ID':<40} {'In $/1M':>9} {'Out $/1M':>10} {'Confirmed':>10}")
        print("─" * 95)
        for m in list_models():
            confirmed = "✓" if m["slug_confirmed"] else "⚠ unverified"
            print(f"{m['key']:<18} {m['model_id']:<40} ${m['pricing']['input']:>7.3f}  ${m['pricing']['output']:>7.3f}  {confirmed:>10}")
        print()
        print("⚠  Unverified slugs are inferred from HuggingFace IDs.")
        print("   Run `python3 models.py probe` to check them against the live /v1/models endpoint.")

    elif cmd == "probe":
        api_key = os.environ.get("PARASAIL_API_KEY", "")
        if not api_key:
            print("PARASAIL_API_KEY not set — cannot probe.")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(base_url="https://api.parasail.io/v1", api_key=api_key)
        print("Fetching live model list from https://api.parasail.io/v1/models ...")
        live = {m.id for m in client.models.list().data}
        print(f"  {len(live)} models available.\n")
        for m in list_models():
            found = m["model_id"] in live
            status = "✓ found" if found else "✗ NOT found"
            print(f"  {m['key']:<18} {m['model_id']:<40} {status}")
        print()
    else:
        print(f"Unknown command: {cmd}. Use 'list' or 'probe'.")
