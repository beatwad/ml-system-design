# YouTube Video Search System

## Business task

Provide users with more relevant video search results → increase user engagement → more ad/revenue.

## Functional requirements (FT)

- Return videos (and only videos) that are the most relevant to user's search text
- Same search result — same output, no user personalization
- Multiple supported languages

## Non-functional requirements (NFT)

- 1 billion videos
- 100 million DAU ~ 30000 RPS (suppose that each user sends 10 requests per day, consider peak load that can be x3 more than usual)
- 10 millions of pairs of video-text train data are available
- p99 latency is 500 ms

## ML task

System must accept text request and return the list of the most relevant video, sorted by decreasing of relevancy — ranking task.

## Data

- Video file
- Video metadata:
    - size
    - timestamp
    - tags
    - short description
    - age restriction
    - language (of audio/transcript, from ASR language detection)
    - duration
    - likes/dislikes
    - views
    - comments
- Text request:
    - timestamp
    - language (detected from query text + user's interface locale)
- Pair text request — link to video

**CLIP-like:** build similarity (logit) matrix of `N x N` where rows correspond to N text requests and columns — to N videos. Pairs on matrix diagonal are positives and marked by 1, the rest are 0. So it's some sort of multiclass classification with N classes.

## Loss function

Contrastive Loss (InfoNCE), symmetric:

- `t_i`, `v_i` — L2-normalized embeddings of text request i and video i in batch of size N
- `sim(t, v)` = cosine similarity = dot product of normalized embeddings
- `temp` — learnable temperature (init ~0.07), kept in log-scale and clamped

```
text -> video: L_t2v(i) = -log( exp(sim(t_i, v_i)/temp) / sum_k exp(sim(t_i, v_k)/temp) )
video -> text: L_v2t(i) = -log( exp(sim(v_i, t_i)/temp) / sum_k exp(sim(v_k, t_i)/temp) )

L = 1/(2N) * sum_i (L_t2v(i) + L_v2t(i))
```

`i, i` — positive pair, all other pairs in batch are negative ones.
Sum in denominator runs over all `k = 1..N` (positive included), otherwise it is not a valid softmax over the batch.

## Model

Learn embeddings using two-tower neural network (bi-encoder):

- First tower accepts:
    - text request (BERT, multilang)
- Second accepts:
    - video embeddings (sequence of frame images, use ViT)
    - text embeddings of video's audio transcript (computed once when uploaded): audio track → ASR (e.g. Whisper) → Long Context BERT → embedding
    - video name + description (BERT, multilang)
    - tags (BERT)

Embeddings from two towers → cosine similarity between embeddings → Contrastive Loss.

Rerank filtered embedding using small cross-encoder BERT-like network (e.g. MiniLM):

- Query (user's request) and documents (video features) are in one sequence and attention has access to all necessary information → that helps to increase model's performance

Final reranking model (LambdaRank) that uses cross-encoder score, popularity, freshness, engagement, language match (request language vs video language), etc. for final model rerank.

Language is a soft feature here, **NOT** a hard filter: multilang encoders match cross-lingual pairs, and user may want content in other language (music, tutorials), so we only boost same-language videos.

## Offline metrics

- In case when multiple videos can be relevant to user's request, we can create Golden Dataset where relevancy of video and text is measured manually in some range (e.g. 1 to 5) and use NDCG
- In case when only one video from search response can be relevant, we use MRR
- Recall@100

Golden Dataset must be stratified by types of queries (head — most popular queries / torso — informational, moderately specific / tail — long, rare entities, misspellings) and languages.
Golden Dataset Pooling: Golden Dataset must be filled with multiple models that are presented in our system (ANN from Video Index Table, ElasticSearch, cross-encoder reranked list, etc). 
This will help to get query results that are as diversed as possible -> recall metric benefits from it.

## Online metrics

- **Long-click CTR** = (number of long clicks) / impressions
    - long click = `watch_time >= max(30 sec, 0.3 * duration)`
    - threshold must depend on duration: fixed 30 sec makes every short video a long click and every 40-min lecture a bounce
    - opposite case is pogo-sticking (click → return back to search page in few seconds) — it is even stronger negative signal than no click at all, because user checked the video and rejected it
- **Watch-time-weighted CTR** = `sum_i min(watch_time_i / duration_i, 1) / impressions`
    - normalization by duration is needed, otherwise metric prefers long videos: 2-hour video watched 10 min would beat 5-min video watched fully
- Average number of minutes/seconds of video watched by users for the last day/week/month
- Like/watch ratio, comment/watch ratio, subscribe/watch ratio
- **Abandonment** = (number of query results where the user clicked nothing) / (total number of query results)
    - careful with "good abandonment": user could get the answer from titles/thumbnails without click
- **Query reformulation rate** = (queries that were edited and re-sent in the same session) / (all queries)
    - stronger dissatisfaction signal than abandonment — user explicitly says that results are bad

CTR-like metrics are sensitive but easy to game (clickbait), so long-click CTR is the primary decision metric, and abandonment/reformulation are guardrails that must not degrade: typical bad case is a change that increases CTR and reformulation rate at the same time.

## A/B tests

Randomization unit is user (hash of `user_id`), because queries of one user are correlated.

Ladder of checks, each next step is ~10x more expensive than previous:

1. **Offline**: NDCG / Recall@100 on stratified golden dataset
2. **Off-policy evaluation**: IPS/doubly robust estimate of online CTR from already logged propensities — costs 0 traffic, lets us screen many candidate rankers before A/B
3. **Interleaving (team-draft)**: merge lists of ranker A and B into one list, alternating who takes the next slot, and compare which ranker's items get more long clicks. Comparison is within one query, so query difficulty variance cancels out → 10-100x more sensitive than A/B. But it gives only preference (A better than B), not delta of business metrics
4. **A/B test of the winner**: 1-5% of traffic, 1-2 weeks, primary metric + guardrails
5. **Ramp up**: 5% → 20% → 50% → 100%, check guardrails on each step

Duration is **NOT** limited by statistical power (0.5% relative change of long-click CTR needs ~10^6 queries per arm — few hours at 1% of traffic), but by validity: 1-2 weeks minimum because of novelty effect and day-of-week seasonality of query mix.

Metrics are reported per stratum (head/torso/tail, language), not only aggregated: regression on tail or on low-resource language easily hides inside a positive total number.

### System-specific details

- Cache key **MUST** include experiment arm, otherwise control's cached results are served to treatment users and the experiment measures nothing
- Exploration traffic is excluded from analysis (its randomness is pure noise for the test)
- Cost of experiment depends on which stage is changed: LambdaRank/cross-encoder change is cheap, but bi-encoder change requires a second full embedding index (64 GB PQ + re-embedding job) running in parallel with production → retrieval changes are batched into rare big experiments, ranking changes are tested often
- Long-running holdback (~1% of users on frozen model for a quarter) to catch slow effects that no 2-week A/B can see: popularity feedback loop, cold start of new videos

## Train

- First train bi-encoder
- Then train cross-encoder on hardest pairs that bi-encoder struggles to solve
- Then distill trained cross-encoder into bi-encoder using soft labels
- Repeat until it stops to improve (usually 2-3 rounds)
- Cross-encoder can be additionally distilled from bigger LLMs (offline teacher)
- Train LambdaRank on cross-encoder score + video interaction features (popularity, freshness, engagement, etc.), video interaction features should be computed at impression timestamp to avoid train/inference skew → video interaction features should be logged

    Label is **NOT** a raw click (raw CTR is gameable by clickbait), but graded relevance built from satisfaction signals of the same impression:

    | Grade | Meaning |
    | --- | --- |
    | 0 | impression without click |
    | 1 | pogo-sticking (click and fast return to search page) — worse than no click at all |
    | 2 | click without long watch |
    | 3 | long click (`watch_time >= max(30 sec, 0.3 * duration)`) |
    | 4 | long click + explicit positive action (like, comment, subscribe, repost) |

    Graded labels are required by the objective itself: LambdaRank gradients are weighted by delta NDCG of swapping two documents, and NDCG is defined on graded gains.

Also both models are updated during system work — system collects users requests and actions (like, comment, subscribe, repost) and builds additional labeled data based on that information.

Hard negatives for bi-encoder are mined from ANN top-k (e.g. k=500) of the current bi-encoder. Rank 500 out of 10^9 means the model put it in the top 0.00005% of the corpus — that's a high score, but 500th video in result wasn't shown to almost anyone → video with relatively high score which is negative → hard negative. Then mined candidates are scored by cross-encoder teacher and the ones it rates as relevant are dropped — they are false negatives. Negatives are re-mined on every distillation round, because "hard" is defined by current model.

Batch size N is the number of negatives per positive, so quality of contrastive training grows with N → use large batches (gradient caching if it doesn't fit into memory).

Some videos can be viewed/liked/commented etc. too often so we also add sample weight for each video: the more it's popular, the less probability of it's sampling to train batch.

Clicks are biased not only by popularity but also by position, so for every impression we log propensity `p` — probability that this video was shown at this position by current policy:

```
query_id, video_id, position, p, policy_version, was_explored, click, watch_time
```

Position bias curve for `p` is estimated from randomized exploration traffic (see Inference). Then each train example is weighted by `1/p` (IPW), with clipping to control variance. Logged propensities also allow off-policy evaluation: estimate online CTR of a new ranker offline (IPS/doubly robust) before spending A/B traffic on it.

## Inference

### Indexing pipeline (offline, once per uploaded video — NOT per query)

- Calculate embeddings for each new uploaded video using video encoder from bi-encoder model
- Calculate transcript (audio → ASR) and its embedding, transcript is cached and reused on retrains
- Calculate tags for each new uploaded video using Tag Service
- Store all of them in video index table, together with filter attributes (age restriction, region, language, status) and text fields for ElasticSearch

### Serving pipeline (online, per user query)

- Pre-process user's query (stemming, lemmatization), add filters
- Ask Cache if the result is in it, if cache hit — return the result and don't go further
- Each user's text request is transformed to embedding by encoder from bi-encoder model and key words are also extracted from it
- Find candidates for user request in video index table by search for nearest neighbours for text embedding using ANN/HNSW/PQ/etc.
- Find candidates for user request in video index table by using key words and ElasticSearch (per-language analyzers: tokenization, stemming and stopwords depend on language, so request language defines which analyzer/index is used)
- Both searches are filtered by age restriction (and region/deleted/private status) **INSIDE** the search, not after it: post-filtering of top-K can return almost empty list if filter is selective. ANN needs filtered search support (e.g. filtered HNSW), ElasticSearch does it with filter clause. Age restriction is compared with viewer's account age/settings — this is the only place where request depends on user, ranking itself stays non-personalized
- Merge two ranked lists of candidates, fuse their ranks — RRF (Reciprocal Rank Fusion):

    ```
    score = Σ_lists w_l / (k + rank_i), k ≈ 60
    ```

    Ranks are used instead of scores because cosine (0..1) and BM25 (unbounded, corpus dependent) are not comparable and both drift when index changes. `k` flattens the top, so appearing in **BOTH** lists is worth more than being rank 1 in one of them. Weights `w_l` depend on query type: dense branch is up-weighted for head/torso, lexical one for tail (see notes about head/torso/tail below), then sort by fused rank, select top 200
- Near-duplicate removal (re-uploads, mirrors, same video with another intro): compare candidates by perceptual/content hash of frames. Go through list from top and drop a candidate if it is a duplicate of any already kept one, until 100 candidates are left or all candidates are processed
- Top 100 candidates are sent to cross-encoder
- Final reranking with LambdaRank
- Topical diversification (MMR) on the final ranked list:

    ```
    score = lambda * relevance_score - (1 - lambda) * max_similarity(candidate, already selected)
    ```

- Randomized exploration (on small part of traffic, to collect unbiased train data):
    - ~1% of queries: swap random adjacent pairs in final list → unbiased estimation of position bias curve (how much CTR drops from rank 1 to rank 10 only because of position)
    - ~1-5% of queries: replace 1-2 slots by candidates from deeper part of retrieval list (ranks 100-1000). Explore at positions 6-10, not at position 1 → we get impression and click signal for much smaller UX cost
    - new videos get guaranteed quota of impressions in first days, otherwise their engagement features for LambdaRank stay empty forever (cold start)
    - exploration works only on already filtered pool (never explore into age restricted, region blocked, spam content) and is turned off for navigational head queries, where random result is too visible
    - cost of exploration is measured as separate A/B arm (CTR/watch time delta vs control)
- Search result is returned to user

### Notes

Text preprocessing services in both train and inference use the same libraries to avoid preprocessing skew.
Video Table Index and Video / Text encoders must be versioned to avoid version mismatch skew.

Depending on query type (head/torso/tail) one or another branch works better:

- **Video Embeddings Index (dense)** — works better for head/torso: queries are verbose and semantic, embeddings generalize over synonyms and paraphrases
- **ElasticSearch (lexical)** — works better for tail: rare proper nouns, product codes, exact entity names. Dense embedding blurs a rare token into its semantic neighbourhood and loses it, while BM25 matches the exact term
- **LambdaRank interaction features** (popularity, engagement) are dense for head and empty for tail, so on tail the final ranking relies mostly on cross-encoder relevance

## Latency estimation

- Network ~ 100 ms
- Text preprocess — 10 ms
- Text encoder (batch = 1, one short query of ~15 tokens): compute is negligible (`2 * 100M * 15 = 3 * 10^9 FLOP`), but at batch 1 we are memory bandwidth bound, not compute bound: all weights are read from HBM to do very little math: `100M params * 2 bytes = 200 MB / ~2 TB/s ~ 0.1 ms`. Real cost is dominated by tokenization, kernel launch overhead and RPC ~ 5 ms
- Search in Video Index Table (1B elements) — 40 ms (in parallel with ElasticSearch)
- ElasticSearch (1B elements) — 40 ms (in parallel with Search in Video Index Table)
- Merge and reranking of two candidate lists (~ 0)
- Reranking using cross-encoder (MiniLM-L6: 6 layers * 384 hidden → ~10.6M non-embedding params, embedding table is a lookup and costs no FLOP, so it is not counted):

    ```
    FLOP = 2 * 10.6 * 10^6 * (100 candidates * 256 tokens) = 5.4 * 10^11
    t = 5.4 * 10^11 / (3.12 * 10^14 * 0.1 (MFU)) = 1.7 * 10^-2 ~ 17 ms
    ```

    MFU = 0.1 is taken as **PESSIMISTIC** bound here on purpose: latency budget is written for p99, when batch is not full, cache is cold and GPU is shared. Expected MFU for full batch of 100 sequences is ~0.4 (see Compute estimation), which gives ~4 ms — that value is used for capacity planning, but it would be wrong to plan p99 latency by average-case number
- Reranking with LambdaRank (~ 0)
- Response — 10 ms

**Total: 182 ms** (p99 budget is 500 ms, so ~320 ms is left as headroom on purpose: retries, GC pauses, cold caches and stragglers of ANN shards eat it in bad cases).

## Memory estimation

- Video Index Table (raw fp16 vectors): `10^9 * 256 * 2 ~ 500 GB RAM`
- PQ compression: split 256-dim vector into 64 sub-vectors, 1 byte code each → 64 bytes per video: `10^9 * 64 ~ 64 GB` → fits into 1-2 machines instead of a rack, at the cost of some recall (approximate distances). Exact re-scoring of top candidates by full vectors compensates it
- Sharding: even 64 GB + HNSW graph is sharded by `video_id` into ~8-16 shards for throughput and availability; router does scatter-gather and merges top-K from all shards
- ElasticSearch index: 1B videos with transcripts (~10 KB of text each) ~ 10 TB of raw text, inverted index is of the same order → tens of nodes, this is much bigger than vector index

## Compute estimation (GPU fleet)

### Online (per query, A100 ~ 3.12 * 10^14 FLOP/s fp16)

- **Text encoder**: batch = 1, so it is memory bandwidth bound, not compute bound: `100M params * 2 bytes = 200 MB / ~2 TB/s ~ 0.1 ms` → ~1-5 GPU for whole fleet, negligible
- **Cross-encoder**: `5.4 * 10^11 FLOP` per query (see Latency estimation). Here we take **EXPECTED** MFU = 0.4 (full batch of 100 sequences), not the pessimistic 0.1 from Latency estimation: fleet size is defined by average throughput, not by p99 of one query

    ```
    t_gpu = 5.4 * 10^11 / (3.12 * 10^14 * 0.4) = 4.3 ms of pure GPU time per query
    ```

    Fleet is sized for peak RPS from NFT (30000 RPS, peak factor is already included there): `3 * 10^4 QPS * 4.3 ms = 129 GPU-seconds per second` → ~130 GPU at 100% util, ~215 GPU at safe 60% util (above ~70% util queueing delay grows non-linearly). With pessimistic MFU = 0.1 it would be ~500-900 GPU
- **Cache of head queries** (results are the same for all users, no personalization) with ~50% hit rate cuts this fleet in half → ~110 GPU
- This is why top-100 was chosen instead of top-500 for cross-encoder: cost is linear in number of candidates, 500 candidates ~ 1000+ GPU (millions of $ per year)

### Offline (indexing)

- **Video encoder**: ViT-B/16 ~ 17.6 GFLOP per frame * 16 frames ~ `2.8 * 10^11 FLOP` per video. 1B videos: `2.8 * 10^20 / (3.12 * 10^14 * 0.4) ~ 2.2 * 10^6 s` ~ 600-1000 GPU-hours → cheap, few hours on 100-200 GPU. Can be repeated on each retrain of bi-encoder
- **ASR**: 1B videos * ~10 min ~ `1.7 * 10^8` hours of audio, Whisper ~ 50-100x realtime → ~10^6 GPU-hours for initial backfill of existing corpus (expensive, one time, months). Steady state: ~500 upload-hours per minute → ~300 GPU running continuously

    **IMPORTANT**: ASR result (transcript) is cached per video and **NEVER** recomputed on retrain, only cheap text encoder pass over cached transcript is redone

## Monitoring

### Input / query side

- Query distribution drift (length, language mix, head/torso/tail ratio), zero-result rate, missing feature rate, ASR failure rate and ASR queue backlog

### Model behaviour (no labels needed, works in real time)

- Score distributions of each stage (bi-encoder cosine, cross-encoder, LambdaRank)
- Rank correlation between stages: if LambdaRank starts to override cross-encoder on most queries, one of them is broken or badly calibrated

### Retrieval quality proxies

- ANN recall vs brute force on a sampled set of queries — the most important one, it is the only thing that catches index degradation after merges, rebuilds or PQ retraining
- Share of final results contributed by each branch (dense vs ElasticSearch): RRF hides a dead branch, list still looks full while half of recall is lost
- Number of candidates that survive filters (detects too selective filtering)

### Business metrics (per stratum and per language, not aggregated)

- Long-click CTR, abandonment, reformulation rate

### Infrastructure

- p50/p99 **PER STAGE** (not only end-to-end), timeout rate, cache hit rate, GPU utilization and queue depth, index freshness lag (upload → searchable)

### Silent failures — no errors, only worse results, so they need explicit alerts

- Encoder version != index version: cosine between two different embedding spaces is meaningless. Hard alert, deploy is blocked on it
- ANN recall drop without any change in latency
- One retrieval branch degrades while RRF masks it
- Preprocessing skew: recompute train-path features on sampled production requests and diff distributions — catches the case when someone bumps tokenizer version
- Feature nulls after upstream schema change: LambdaRank silently degrades to ranking by cross-encoder score only, and dashboards look fine

## Fallback (graceful degradation)

Main principle: every stage has its own deadline and a **DEFINED** degraded output. Worse list is always better than an error page, and the deeper the failed stage is, the smaller the loss.

- **Text encoder / GPU unavailable** → serve ElasticSearch-only results. Tail queries almost don't suffer, head/torso lose semantic generalization
- **ANN index unavailable** → ElasticSearch-only. Partial shard failure: scatter-gather returns what came back before deadline (accept K of N shards) and logs shard coverage
- **ElasticSearch unavailable** → dense-only results, tail queries suffer most
- **Both retrieval branches down** → last resort: precomputed static top list for head queries or stale cache with relaxed TTL
- **Cross-encoder timeout or GPU fleet saturated** → return the RRF-fused order as is. This is the cheapest and most useful fallback, because cross-encoder is the expensive stage. Better than binary on/off: under overload reduce candidates 100 → 25 (load shedding)
- **LambdaRank or feature store unavailable** → rank by cross-encoder score only. Must be an **EXPLICIT** branch, otherwise feature nulls silently degrade ranking (see Monitoring)
- **Dedup / MMR failure** → skip, serve list as is (cosmetic degradation)

### Two special cases

- Cache is not only a latency optimization but a **CAPACITY** dependency: it serves ~50% of traffic, so if it dies, demand on GPU fleet doubles instantly → cascading failure. Either plan fleet capacity without cache, or shed load (fewer candidates, drop exploration first) when cache hit rate falls
- Filters (age restriction, region, deleted/private) **FAIL CLOSED**, not open: everything else degrades by "serve something worse", but here we must never serve unfiltered results. If filter attributes are unavailable — serve only content without any restrictions

### Cross-cutting

- Deadline propagation: each stage gets the **REMAINING** budget, not a fixed timeout
- Circuit breakers: after N consecutive failures stop calling the stage for T seconds, otherwise queues pile up and the whole pipeline slows down
- Retries only once, with jitter: retry storm under overload makes things worse
- Encoder/index version mismatch → block deploy and roll back to previous consistent pair
- "Degraded mode rate" is itself a monitored metric with an alert

## TODO

- Use an ML model to find titles and tags that are semantically similar to text queries. This model can be combined with ElasticSearch to improve search quality.
- An important aspect of search systems is query understanding: the system corrects spelling, identifies the query category and recognizes entities. How to build a component that would analyze the meaning of a query?
