# 03 — Unit economics of LLM products

All prices below are **placeholders**. Plug in current provider prices and your measured token counts.

## Cost per task
`cost = Σ_calls (input_tokens × p_in + output_tokens × p_out) − cache discounts + retrieval + infra`
Reasoning models add hidden "thinking" tokens billed as output, so measure them. Example: a document extraction task with 6k input tokens (the pages) and 800 output tokens, at $3/M in and $15/M out: $0.018 + $0.012 = **$0.030/document**. At 50k documents/month that is $1,500. If a human costs ₹15/document (~$0.18), you can price at $0.08–0.10 and keep ~65–70% gross margin.

## Levers, roughly in order of impact
1. **Fewer tokens:** crop inputs (retrieve the relevant pages), shorter prompts, prompt caching for stable prefixes, structured outputs.
2. **Routing/cascades:** a cheap model first, escalate on low confidence. Expected cost = `c_small + P(escalate)·c_large`. Measure P(escalate) and the quality delta with evals.
3. **Batch APIs** for offline work (often ~50% cheaper).
4. **Fine-tune a small model** once volume is high and the task is stable.
5. **Self-host:** `$/M tokens = GPU $/hour ÷ (tokens/s × 3600) × 10⁶`. At $2/hour and 1,500 tokens/s aggregate that is $0.37/M at *full* utilization; at 30% utilization, $1.23/M. Utilization is the business ([lab 03](../../labs/03_napkin_math/README.md)). Self-host when volume is steady, when privacy requires it, or when a small fine-tuned model beats the API on your evals.

## SaaS basics you must know cold
Gross margin = (revenue − COGS)/revenue; AI apps often sit at 50–70% vs ~80% for classic SaaS. **CAC** (sales and marketing ÷ new customers); **LTV** ≈ ARPA × gross margin ÷ monthly churn; LTV/CAC > 3; CAC payback < 12–18 months. Burn and runway; "default alive?" (Paul Graham): at the current growth and burn, do you reach profitability before the money runs out?

## Pricing
Per seat (simple, but misaligned with AI's value), usage-based (aligned with cost), or **outcome-based** (per document processed, per ticket resolved), which is often best for automation because it prices against the labor you replace. Never run free pilots indefinitely: charge something, with success criteria written into the pilot.
