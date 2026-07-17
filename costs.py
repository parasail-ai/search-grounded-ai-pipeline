"""
costs.py — Query cost calculator for Parasail + You.com Search.

Adapted from comparison/compare.py in the grounding repo. Kept distinct so
pricing logic stays auditable and testable independently of the server.

You.com Search pricing: $5 per 1,000 queries = $0.005 per call.
Livecrawl overage: first 10 URLs/call included; extra URLs at $1/1,000.

Usage:
    from costs import calculate_costs
    costs = calculate_costs(stats, model_pricing)
    # → {"llm": 0.0012, "search": 0.005, "livecrawl_overage": 0.0, "total": 0.0062}
"""

YDC_SEARCH_COST_PER_CALL = 0.005   # $5 per 1,000 queries
YDC_LIVECRAWL_OVERAGE_PER_URL = 0.001  # $1 per 1,000 extra URLs → $0.001/URL


def calculate_costs(stats: dict, model_pricing: dict) -> dict:
    """Return {"llm", "search", "livecrawl_overage", "total"} in USD.

    Args:
        stats: agent stats dict with keys:
            - token_breakdown: {"input": int, "output": int, ...}
            - search_calls: int
            - tool_calls: list of {"livecrawl": bool, "count": int, ...}
        model_pricing: dict with keys:
            - input:  price per 1M input tokens (USD)
            - output: price per 1M output tokens (USD)
    """
    tb = stats.get("token_breakdown", {})
    input_tokens  = tb.get("input",  stats.get("input_tokens",  0))
    output_tokens = tb.get("output", stats.get("output_tokens", 0))

    llm = (
        input_tokens  * model_pricing.get("input",  0) / 1_000_000
        + output_tokens * model_pricing.get("output", 0) / 1_000_000
    )

    search = stats.get("search_calls", 0) * YDC_SEARCH_COST_PER_CALL

    livecrawl_overage = 0.0
    for call in stats.get("tool_calls", []):
        if call.get("livecrawl"):
            count = call.get("count", 5)
            excess = max(0, count - 10)
            livecrawl_overage += excess * YDC_LIVECRAWL_OVERAGE_PER_URL

    total = llm + search + livecrawl_overage
    return {
        "llm": llm,
        "search": search,
        "livecrawl_overage": livecrawl_overage,
        "total": total,
    }


def format_cost(amount: float) -> str:
    """Human-readable cost string. Switches precision based on magnitude."""
    if amount == 0:
        return "$0.000000"
    if amount < 0.001:
        return f"${amount:.6f}"
    if amount < 0.01:
        return f"${amount:.5f}"
    return f"${amount:.4f}"


# ─── CLI self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_stats = {
        "token_breakdown": {"input": 10_000, "output": 500},
        "search_calls": 3,
        "tool_calls": [
            {"livecrawl": True, "count": 5},
            {"livecrawl": True, "count": 15},  # 5 excess URLs
            {"livecrawl": False, "count": 5},
        ],
    }
    pricing = {"input": 0.04, "output": 0.20}  # GPT-OSS 20B
    c = calculate_costs(sample_stats, pricing)
    print(f"LLM cost:            {format_cost(c['llm'])}")
    print(f"Search cost:         {format_cost(c['search'])}")
    print(f"Livecrawl overage:   {format_cost(c['livecrawl_overage'])}")
    print(f"Total:               {format_cost(c['total'])}")
