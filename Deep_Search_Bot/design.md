# Deep Search Bot (Agentic Multi-Source Research over Web + Internal Knowledge Bases)

## Business task

Speed up information search process -> increase employees performance -> increase the number of tasks fullfiled by employees per day/week/month.

## Functional requirements (FT)

- Decompose a question into sub-questions, iteratively researching each with the right source/tool
- Sources: public web (search API), Jira, Confluence, Slack, and SQL databases — ClickHouse (analytics/events) and Postgres (application/transactional data), extensible to other databases via the same connector pattern — queried through natural-language-to-SQL
- Return a structured, cited report — every claim traceable to a source (URL, Jira ticket, Confluence page, Slack thread, or the SQL query + result that produced a number)
- Support multi-hop questions where a later sub-question depends on an earlier answer (e.g., "which service had the most incidents last quarter, and what were the top root causes in their postmortems?")
- Self-assess sufficiency: keep researching until evidence is enough, or a budget is exhausted — then state what's unresolved instead of guessing
- Respect per-source permissions: Jira/Confluence/Slack ACLs (Access Control Lists), database roles/row-level access
- Long-running async job: submit → stream progress updates → deliver final report

## Non-functional requirements (NFT)

- Low volume, high cost per query: ~2,000 employees with access, ~200 DAU (Daily Active Users), ~1.5 queries/user/day → ~300 queries/day (~0.003 QPS avg) — constraint is cost and per-query latency, not throughput
- Internal corpus: ~3M documents across Jira/Confluence/Slack → ~12M chunks
- Databases: existing ClickHouse and Postgres clusters, queried live, never indexed/copied
- Freshness: internal index lag ~15 min SLA (Service-Level Agreement); web and databases are queried live
- Latency: p99 time-to-first-progress-update < 2 s, p99 time-to-full-report < 90 s, hard cap 5 min/job
- Cost per query is a first-class metric — multiple LLM (Large Language Model) calls plus external API calls per job, well above single-shot RAG

## ML task

An agentic pipeline chaining several ML tasks in a loop:

1. **Query decomposition / planning** — question → ordered sub-questions
2. **Tool selection / routing** — sub-question → source (web / Jira+Confluence / Slack / SQL database), via LLM function-calling
3. **Per-source retrieval** — dense + lexical + rerank per internal source, plus web search + page fetch/parse
4. **Text-to-SQL (semantic parsing)** — sub-question → dialect-specific SQL grounded in the target database's schema
5. **Sufficiency judgment (reflection)** — does the evidence so far answer the question, or is another iteration needed?
6. **Grounded multi-source synthesis** — evidence pool → one coherent cited report

## Data

- **Web**: fetched live via search API + page fetch/parse — transient per job, never pre-indexed
- **Jira**: issue/comment text, project, status, assignee, timestamps, ACL
- **Confluence**: page text, space, author, timestamps, ACL — chunked/embedded like Jira
- **Slack**: message/thread text, channel, timestamp, ACL from channel membership
- **SQL databases**: not the row data — schema (tables, columns/types, short descriptions, sample rows) per database and dialect, used to ground SQL generation. Query execution is live and read-only
- **Job trace logs**: query, sub-questions, per-iteration (tool, input, output, latency, error), evidence pool with citations, final report, thumbs up/down, citation clicks — audit trail and training data
- **Golden eval set**: SME (Subject Matter Expert)-curated multi-hop questions spanning ≥2 source types, with a reference report and minimal source set — refreshed as docs/schemas change
- **Text-to-SQL training pairs**: (question, gold SQL, expected result) per table/dialect, plus synthetic augmentation

## Model

### Why an agentic loop instead of single-shot RAG

- Single-shot RAG works when one query embedding locates the answer in one pass. It breaks on questions that require *composing* facts across sources no single retrieval call would surface together (a Jira ticket, a Confluence postmortem, a database query confirming impact)
- An agentic loop treats the LLM as a **planner that inspects intermediate results and decides what to look up next** — closer to how a person researches
- Trade-off: latency/cost scale with iteration count, so every iteration is bounded (max sub-questions, max tool calls, wall-clock cap), and reflection exists specifically to stop as soon as evidence is sufficient

### Planner

An LLM call produces an initial ordered sub-question list, each optionally tagged with a suggested source. The plan is revised as evidence comes in — sub-question 1 (e.g., "which service had the most incidents") is often a prerequisite for phrasing sub-question 2 correctly, so decomposition isn't fixed upfront.

### Tool router and tools

Each iteration, an LLM function-calling step picks the next sub-question and tool:

- **Web search**: search API → top results → fetch + parse → chunk and rerank the same way as internal documents, so only the relevant passage enters the evidence pool
- **Jira/Confluence**: hybrid dense (ANN (Approximate Nearest Neighbor)/HNSW (Hierarchical Navigable Small World)) + lexical (BM25) retrieval fused with Reciprocal Rank Fusion, then cross-encoder reranked; the ACL filter runs inside both searches, not after, and fails closed on missing ACL metadata
- **Slack**: message search API scoped to channels the requesting user can see
- **SQL database (Text-to-SQL)**: picks the target database from the sub-question's domain (transactional/product data → Postgres, analytics/event aggregates → ClickHouse), generates dialect-specific SQL against its schema, then a **safety gate** — read-only role, table allowlist, `EXPLAIN`/dry-run cost check, row/time limits, query timeout — before execution. Non-negotiable: this is the one tool that could mutate state or run an unbounded-cost query if ungated

### Reflection / sufficiency judgment

After each iteration, an LLM call checks the evidence pool against the original question: enough to answer, or what's the specific gap? This lets the loop surface sub-questions the initial plan missed. It exits when reflection says "sufficient," or when the budget runs out (report still produced, marked partial).

### Synthesis

One LLM call over the full evidence pool (each item tagged with source and citation metadata) produces a structured report — an answer per sub-question plus a summary — with every claim citing its source (URL, ticket, page, thread, or SQL query + result).

### Grounding / hallucination and prompt-injection mitigation

- Prompt-level constraint ("only use the provided evidence") plus an async post-hoc groundedness check (NLI (Natural Language Inference) model or LLM-judge verifying each claim is entailed by its cited evidence) — sampled after delivery, feeds monitoring rather than blocking the response
- **Fetched web content is untrusted**: pages can contain text engineered to look like instructions. Fetched content is wrapped and labeled as data, never concatenated as an instruction, and a lightweight injection-pattern check flags suspicious pages before they enter the evidence pool
- SQL results count as ground truth only if the query passed the safety gate — a failed/timed-out query is surfaced as a gap, never approximated

## Loss function

Most components are a **pretrained instruction-tuned LLM used via prompting/function-calling**, not trained — effort concentrates on two components with a clean supervised signal:

- **Per-source rerankers** (Jira/Confluence/Slack): bi-encoder with symmetric InfoNCE (Info Noise-Contrastive Estimation) contrastive loss for retrieval, cross-encoder reranker with pointwise binary cross-entropy on labeled (query, chunk, relevant?) pairs
- **Text-to-SQL model**: fine-tuned code-generation model (cross-entropy over the SQL token sequence, conditioned on question + schema + target dialect) or a pretrained LLM via in-context learning with schema in the prompt. Fine-tuning pays off because dialect-specific quirks (ClickHouse array/map functions and `FINAL`, Postgres CTEs (Common Table Expressions) and window functions) are underrepresented in general pretraining data
- **Planner, router, reflection, synthesizer**: **not trained** — prompted calls to a pretrained LLM with function-calling. Training a full agent end-to-end (RL (Reinforcement Learning) over trajectories, or DPO (Direct Preference Optimization) comparing efficient vs. wasteful paths) is expensive and unstable; prompting a strong model captures most of the value cheaply. Optional later step: SFT (Supervised Fine-Tuning)/DPO on curated trajectories from job-trace logs, to cut wasted iterations — never to inject facts

## Offline metrics

- Per-source retrieval: Recall@k/NDCG (Normalized Discounted Cumulative Gain)@k/MRR (Mean Reciprocal Rank), evaluated separately per source
- Text-to-SQL: execution accuracy (result match, not string match) per database/dialect on the golden set
- Decomposition coverage against golden reference sources
- Tool selection accuracy: did the router pick a source that could answer the sub-question
- End-to-end: faithfulness/groundedness, citation accuracy per source type, and **iteration efficiency** — loop iterations used vs. the minimum needed (cost proxy)

## Online metrics

- Thumbs up/down on the report
- Escalation rate — user still does the research manually or asks a human afterward
- Cost per query (LLM tokens + external API calls) — a primary metric, not just infra dashboard noise
- Time-to-full-report distribution and partial-report rate (budget hit before reflection said "sufficient")
- Citation click-through per source type

## Train

- Build the golden multi-hop eval set first — nothing else can be judged without it
- Train per-source bi-encoders/rerankers with InfoNCE + hard-negative mining, one pass per source (query patterns differ: Slack is short/informal, Confluence is long/formal)
- Fine-tune Text-to-SQL per database/dialect on curated + synthetic triples; re-run on material schema changes
- Planner/router/reflection/synthesizer stay prompt-only initially; job-trace logs (thumbs-down + escalations, human-reviewed) feed an optional later SFT/DPO pass to cut wasted iterations, not to change correctness

## Inference

### Indexing (offline/continuous)

- Jira/Confluence/Slack: connector (webhook or periodic sync) → extract → chunk (semantic, heading-aware, ~15% overlap) → embed → hybrid vector + lexical index, ACL resolved from the parent doc; ~15 min freshness SLA
- SQL databases: only schema/metadata is indexed for grounding, refreshed on schema change (rare relative to row data changing)

### Serving pipeline (async job, per query)

1. Authenticate user, resolve permission groups (ACLs + database roles)
2. Create job, return job id immediately; stream progress events
3. Planner LLM call → initial sub-questions
4. Loop (bounded by max iterations and wall-clock budget):
    1. Tool router picks sub-question + tool
    2. Execute (web fetch, ACL-filtered internal retrieval, or Text-to-SQL with the safety gate), stream a progress event
    3. Append result + citation metadata to the evidence pool
    4. Reflection call: sufficient? if not, what's missing? → continue or break
5. Synthesizer call → structured cited report (marked partial if budget-exited)
6. Async: sample into the groundedness/eval pipeline
7. Deliver report; persist the full job trace for the training/eval flywheel

### Notes

Every tool call is logged with exact input/output — this makes the job trace debuggable and directly reusable as training data.

## A/B tests

- Randomization unit: user
- Ladder, cheap → expensive:
    1. Offline: per-source Recall@k, Text-to-SQL execution accuracy, decomposition coverage, iteration efficiency
    2. Online, small traffic slice: primary metric = thumbs-up rate, guardrails = escalation rate, partial-report rate, cost per query, sampled human-reviewed hallucination rate
    3. Ramp up with guardrails checked at each step; cost per query watched closely, since a planner/prompt change can silently increase average iterations
- Report per team/question-type — a regression isolated to one tool (e.g., Text-to-SQL) is easy to miss in an aggregate number

## Monitoring

- Per-tool error/timeout rate — a silently-failing source degrades coverage without an obvious error
- Loop iteration count distribution — a shift upward signals a cost/latency regression
- Partial-report rate — leading indicator the budget doesn't match real question complexity
- SQL safety-gate rejection rate and reasons, tracked separately from execution failures
- Prompt-injection flag rate on fetched web content
- Cost per query, tracked with the same seriousness as latency
- Permission-leak canary — scheduled queries from low-permission synthetic accounts, asserting restricted content never appears in evidence or citations; a silent-failure class that needs a hard alert, not a dashboard

## Fallback (graceful degradation)

- One tool/source down → skip it, continue with the rest, note the gap in the final report
- A database unavailable → its quantitative sub-questions marked unresolved, never estimated from prose sources
- Web search unavailable → answer from internal sources only, report notes the gap
- Reflection unavailable → fall back to the full initial plan (no early exit or gap-filling), still bounded by the iteration cap
- SQL safety gate rejects every candidate query → treated as a tool failure, never relaxed
- Permission service unavailable for a tool → fail closed for that tool only, rest of the loop continues

## Latency estimation

Dominated by iteration count and by *external, non-compute-bound* calls (web fetch, SQL execution) rather than FLOP (floating-point operations)-estimable compute.

- Planner call (small/fast LLM, short output): ~800 ms–1 s
- Per iteration (~3 typical):
    - Tool router call: ~300 ms
    - Tool execution — dominant, variable term:
        - Web: search API (~500-800 ms) + parallel fetch/parse with timeout (~1.5-2 s) ≈ 2-2.5 s
        - Internal RAG tools: ~50-100 ms (in-cluster, ~12M chunks)
        - SQL database: Text-to-SQL generation (~500 ms) + safety check (~50 ms) + execution (~1-2 s for a well-formed aggregate query)
    - Reflection call: ~500-800 ms
    - → ~2-3.5 s/iteration, mostly web/SQL-bound
- Synthesis call (large context, ~800-1500 output tokens): 70B-parameter model tensor-parallel across 8 A100s (GPU (Graphics Processing Unit)s). Prefill on a ~3,000-token prompt: `2*70e9*3000 / (8*3.12e14*0.4 MFU (Model FLOP Utilization)) ≈ 0.45 s`. Decode, memory-bandwidth bound: `140 GB fp8 weights / 16 TB/s aggregate HBM (High Bandwidth Memory) ≈ 8.75 ms/token` → 1500 tok ≈ 13 s
- **Typical (3 iterations): ~1 + 3*3 + 13 ≈ 23 s** — within the 90 s p99 budget
- **Worst case** (5 iterations, web-heavy, slow SQL): ~1 + 5*3.5 + 13 ≈ 31.5 s, with tail cases pushing toward the 5 min hard cap — the reason that cap exists independent of the iteration cap
- Time-to-first-progress-update ≈ the planner call, ~1 s — well under the 2 s target

## Memory estimation

- Jira/Confluence/Slack indices: ~12M chunks * 768-dim * 2 bytes (fp16) ≈ 18 GB raw; PQ (Product Quantization)-compressed (96 sub-vectors, 1 byte each) → ~1.2 GB; lexical index a low double-digit GB — fits on a handful of nodes
- SQL database schema/metadata index: a few MB, trivial
- LLM weights: 70B params, 1 byte/param (fp8) ≈ 70 GB + KV (Key-Value)-cache — the same model instance serves varying-length calls per job (planner, router, reflection ×N, synthesis)

## Compute estimation (GPU fleet)

- Query volume is very low (~300/day, ~0.003 QPS avg) — sized by **total LLM-seconds per query**, not raw QPS: each job makes ~1 (planner) + 3×2 (router+reflection) + 1 (synthesis) ≈ 8 LLM calls
- At ~23 s of LLM-active time per typical job and ~300 jobs/day, peak concurrent LLM-occupied jobs is small (low single digits) — a single 8-GPU node with continuous batching has large headroom
- At this utilization, a dedicated self-hosted fleet is hard to justify — evaluate a hosted LLM API (zero-data-retention agreement) as build-vs-buy, since idle GPU time is the dominant cost risk, not per-token pricing
- The real cost driver at this volume is **external API cost** (web search, ~2-3 calls/job) and database query load, not GPU fleet size
