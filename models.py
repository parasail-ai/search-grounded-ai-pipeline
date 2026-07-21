"""
models.py — Parasail model registry with pricing.

Pricing source: parasail.io/pricing (per 1M tokens, serverless tier).
API slugs: confirmed via docs or GET https://api.parasail.io/v1/models.

The registry contains the Parasail model IDs used by the demo.
Verify current slugs by running: python3 models.py list

Usage:
    from models import MODELS, get_model, default_model

    cfg = get_model("deepseek-v4-pro")
    agent = ParasailAgent(model=cfg["model_id"])
"""

# ─── Model registry ──────────────────────────────────────────────────────────
# Keys are short display names used in the UI / CLI.
# model_id: the string to pass to the API as "model"
# Pricing: per 1M tokens (serverless tier), USD.
# slug_confirmed: True = verified via API or official docs; False = inferred from HF ID, verify via /v1/models

PARASAIL_BASE_URL = "https://api.parasail.io/v1"

MODELS = {
    # ── GLM (Zhipu AI) ────────────────────────────────────────────────────────
    "glm-5.2": {
        "model_id": "zai-org/GLM-5.2",
        "display_name": "GLM 5.2",
        "provider": "Zhipu AI",
        "pricing": {"input": 1.40, "output": 4.40, "cache_read": 0.26},
        "slug_confirmed": True,
    },

    # ── MiniMax ──────────────────────────────────────────────────────────────
    "minimax-m3": {
        "model_id": "MiniMaxAI/MiniMax-M3",
        "display_name": "MiniMax M3",
        "provider": "MiniMax",
        "pricing": {"input": 0.30, "output": 1.20, "cache_read": 0.06},
        "slug_confirmed": True,
    },

    # ── Kimi (Moonshot AI) ────────────────────────────────────────────────────
    "kimi-k2.7-code": {
        "model_id": "moonshotai/Kimi-K2.7-Code",
        "display_name": "Kimi K2.7 Code",
        "provider": "Moonshot AI",
        "pricing": {"input": 0.75, "output": 3.50, "cache_read": 0.16},
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
DEFAULT_MODEL = "deepseek-v4-pro"


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
        base_url=PARASAIL_BASE_URL,
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
