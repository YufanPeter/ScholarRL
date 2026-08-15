"""The search environment: the agent acts, the env retrieves and observes.

Action contract (rigid tags; small models need this):
    <think>...</think>
    <search> query </search>          -> <observation> top-k titles+abstracts </observation>
    <read> paper_id </read>           -> that paper's abstract
    <select> paper_id, ... </select>  -> accumulate answer set
    <finish/>                         -> terminate

Env rules: max 3-4 turns, dedup repeated queries, top-3 results, truncate abstracts,
parse failure -> format penalty (never crash).
"""
