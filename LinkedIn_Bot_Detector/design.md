# LinkedIn Bot Detector

## Business Task
Detect bot behaviour and block them to prevent their interaction with users. The business task is to increase user's engagement and loyalty to the platform -> increase profit from subscription and ads.

## FT
- detect bot-like behaviour 
- the detection trigger is a suspicious behaviour - a lot of complains on user or a lot of complains FROM user and users with whom the account is connected or shares same IPs/subnets, connection from the unusual location (e.g. different country), using VPN and proxies, a lot of actions for some period of time, mass DMs, job applies or connection requests etc.
- in case of high confidence block the corresponding account automatically
- in case of mid confidence send the account a challenge (CAPTCHA, SMS/phone verification, KYC) and/or softly limit it (rate limit on DMs/connection requests, reach throttling)
- only accounts that failed the challenge are sent to the Operator

## NFT
- 1 billion users
- DAU 100 millions
- Suppose that 1 of 10 users demonstrates suspicious behaviour every day RPS: 10^8 / (10 * 10^5) = 1000 RPS
- Number of checks per day: 1000 RPS * 10^5 = ~10^8 checks/day
- FP rate < 10^-5 for the auto-block threshold -> <= 1k false user blocks/day
- Mid-confidence cases go to Operators, so the second threshold is set by the daily review capacity, not by the FP budget

## ML task
Binary classification task with class imbalance handling

## Offline metric
PR-AUC
Fix the Specificity (0.99999, see the FP budget in NFT) and increase Recall for that specificity
Both of those metrics are sustainable to a huge class imbalance
Also control Precision exactly at the auto-block threshold.

## Online metric
Number of user complains to bot-like behaviour
Number of user appeals to incorrect blocks
Churn rate
Time that user spends in application
User activity: likes, comments, reposts, etc.
Number of paid subscription

## Data
- user features
    - user_id
    - creation_date
    - country
    - city
    - language
    - is_blocked
    - last_checked
    - last_activity
    - list of connections
    - list of used IPs
    - is_vpn
    - device fingerprint (device_id, OS, browser, screen resolution, timezone, locale)
    - registration data: IP/ASN, email domain, phone prefix, referrer, invite code
- history of user blocks:
    - timestamp_blocked
    - timestamp_unblocked
    - reason
- history of user activity:
    - activity_id
    - timestamp
    - type of activity (click, post, like, comment, view, article, etc)
    - text of post/comment/article
    - complain list:
        - complain reason
        - how many users complained on user's activity
    - additional features for post/comment/article: CTR, likes, views, comments, tags, etc
- additional features:
    - avg time between loading page and action
    - avg number of activities per hour
    - distribution of activities by hours
    - text sentiment analysis (main theme: Crypto, Politics, Sport, etc.)
    - text links - are sites where they leads in list of scam sites?
- account-linkage features (link accounts that share an identity artifact - device_id, IP/ASN, email domain, phone prefix, payment token, near-duplicate profile text/photo):
    - size of the linked account cluster
    - share of already blocked accounts in the cluster
    - how many accounts of the cluster were registered in the same short time window (burst registration)
    - age of the cluster, share of accounts with empty/duplicate profile
- graph features (connection graph + interaction graph "who liked/commented/DM-ed whom"):
    - in/out degree, ratio of sent connection requests to accepted ones
    - local clustering coefficient, number of connected components in the neighbourhood (bots connect to random unrelated users -> low coefficient)
    - share of neighbours that are blocked (at the time when suspicious behaviour was detected) or belong to the same linked cluster
    - PageRank / k-core position - real accounts are embedded in dense professional communities, farm accounts sit on the periphery
    - coordination features: number of other accounts that act on the same targets/posts within a small time window, similarity of activity time distribution to other accounts of the cluster
    - node2vec/GraphSAGE embedding of the user as a dense feature

Data example: user features + blocking history + last 50-100 activities + list of connections and how many of them were blocked for bot-like behaviour + linkage/graph features + additional features

All linkage/graph/behavioural aggregates are computed as of detection timestamp, from a single feature store. 

## Model
- Two approaches: Baseline GBDT, then can switch to BERT-like transformer, which is good for sequence processing.
- Graph part: start with hand-crafted linkage/graph aggregates as features for GBDT (cheap, recomputed in batch), then can switch to GNN (Graph Neural Network, e.g. GraphSAGE) over the connection/interaction graph and feed its embedding into the main model.
- Cluster-level decision: score the linked account cluster as a whole, not only the single account - if a large share of the cluster is confidently bot-like, raise the score of the remaining accounts (a single account is easy to disguise, a farm of 10k is not).

Fine-tune the threshold based on offline metric

## Train
Train/val/split based on time, stratify by bot campaings (bots from the same bot campaing must belong to either train or val/test datasets), optimize BCE.

How to define that user is bot? He/she was blocked for bot-like behaviour and didn't appeal his/her blocking during some time (e.g. one month) or appelation was failed (e.g. failed to KYC) and no additional attempts were made in one month. This bot accounts must be additionally verified by SMEs to prevent false positives.

Additionally add users that were blocked but they are not the bots, and bots that behave like humans (hard negatives and hard positives).

Problem: labels defined this way are produced by our own blocking system, so the model learns to repeat the current triggers and we can't see the bots it misses (no way to estimate the real Recall). 
Two additional label sources:
- Enforcement holdback: a small share of high-score accounts (e.g. 0.5-1%) is scored but NOT blocked, only logged. We keep watching them and see what they do next - it gives unbiased Precision of the auto-block threshold and shows the harm we prevent.
- Honeypots: our own accounts/posts/job openings that a real user has no reason to interact with (hidden profiles, fake vacancies, non-clickable links). Interaction with them is a clean positive label that doesn't depend on our blocker.
- Also add a regular random audit: SMEs review a random sample of blocked accounts, this gives the true FP rate (appeals alone underestimate it - real bots never appeal).

Retrain should be conducted weekly or bi-weekly or when monitoring trigger fires (look Monitoring paragrpaph).

## Inference
- Detection Service detects the suspicious user behaviour
- It collects user data and put them into Queue
- Then model takes the user data from the Queue
- Model returns the score:
    - If it's high - user is blocked automatically (can appeal this later)
    - If it's mid - the account gets a challenge (CAPTCHA, SMS/phone verification, KYC) and soft limits (rate limit on DMs/connection requests, reach throttling) until the challenge is passed
    - If the challenge is failed or ignored during some time - the account goes to the Operator queue
The auto-block threshold must be tight, according to the FP budget from NFT.

The operator and appeal loop are out of scope of this design.

Additionally can add service that checks every account (Sweep Detector Service) just by scanning the User database looking for users that weren't checked for some time.

How to stop mass complains:
- cap the max complains number (e.g. no more than 10 per hour)
- mass complains is a suspicious behaviour too

How to cold start: 
- we should have in our dataset an accounts without a history which belong to bots and users. Long-term history features can be nulls or group means (group by country/city/age/gender etc.). And also despite the absense of long-term history, we can still collect short-term history of user activity for these couple of days (latest clicks, posts, time between loading page and action, number of activities per hour, etc.). These features (+ small account age) are enough to detect the bot if it acts like a bot.

We also should cap the max number of accounts from one cluster that can be blocked at one time to prevent damage to an accounts that use corporate NAT or university network. Also exclude known shared-infrastructure ASNs. ASN = Autonomous System Number — the ID of the network block an IP address belongs to (e.g. AS15169 = Google, AS16509 = AWS, AS8075 = Microsoft). 

## A/B tests + montoring
At first check offline metrics

The problem in A/B test of this system is a leakage of bots from control group to ordinary group because each user/bot from control group can be connected with a lot of users/bots from ordinary group.

So we run the model online and measure model quality (Precision/Recall of the auto-block threshold) using enforcement holdback (a small random slice of accounts your model scores above the auto-block threshold that you deliberately don't block) - we can evaluate which accounts from this holdback are really bots by analizing their further behaviour and what damage they can cause. 

Also randomly audit blocked accounts using SMEs to find out the False Positives.

Ecosystem metrics:
    - Avg time that user spends in application
    - User activity
    - Number of bought subscriptions
    - Number of complains/appeals/churn rate

They are measured by geo-randomization (enforcement is enabled in some countries/regions and disabled in others) or by switchback (turn enforcement on/off for the whole platform by time intervals) - the group is a closed ecosystem and bots don't leak into the control.

## Monitoring
- Number of complains, appeals and churn rate
- Score-distribution drift
- Per-country/language FP rate
- Operator queue depth
- Feature freshness/null rate
- Honeypot catch rate

 High score-distribution drift or low honeypot catch rate triggers retrain process.

## Fallback
If the specific model version fails or demonstrates degraded performance - roll back to the previous version.

If the model is fully unavailable - switch to the degraded mode:
- a static rules engine works instead of the model (the same triggers that the Detection Service uses: rate of DMs/connection requests, complains, datacenter ASN/VPN, burst registration, share of blocked accounts in the cluster). It is much less accurate, so it MUST NOT block automatically - it only issues challenges and soft limits.
- auto-block is disabled while the model is down
- operators keep working on the accounts that failed the challenge and on the accounts that were complained by, in priority of the expected damage

What happens to the queue:
- messages are not dropped, the Queue keeps them (retention e.g. 24h) and they are replayed when the model is back
- the queue is prioritized: complains and high-risk triggers first, the Sweep Detector Service traffic last (it can be paused at all to save the throughput)
- if the backlog grows over the retention/SLA - drop the oldest sweep-generated messages first, these users will be re-checked by the next sweep anyway

## Compute
Online:
- GBDT model is almost free: 500 trees x depth 8 = ~4k comparisons, ~5 us per user. At 1000 RPS = 5 ms of CPU per second, i.e. <1% of one core. Model size T * 2^D * ~24B = ~3 MB - fits in L2/L3 cache, so it stays fast and is simply replicated in every pod.
- Feature store lookups are the main online cost, but they are I/O-bound (network + KV storage), not CPU-bound: 10-50 ms of waiting, not of computation. So we scale by concurrency, not by cores: 1000 RPS * 50 ms = ~50 requests in flight. 10-20 CPUs are enough for 1000 RPS (mostly for deserialization and feature assembly), scale to ~50 CPU for peaks (mass complains on a botnet). The feature store itself needs to hold 1000 RPS * ~200 keys = ~2*10^5 lookups/sec - a sharded Redis/Cassandra cluster.

Offline (this is where the real money is spent):
- Graph jobs are the dominant cost. 10^9 nodes, ~100 connections per user => ~10^11 edges. PageRank/k-core is O(E) per iteration, ~20 iterations => ~2*10^12 edge operations per full recompute; connected components over the linkage graph is comparable. That is a Spark/GraphX cluster of ~100-200 cores for a few hours. Run it daily, not per-request.
- Cheap graph aggregates (degree, accept ratio, share of blocked neighbours, coordination counters) are updated incrementally in a streaming job, so they don't wait for the daily batch.
- GBDT training: ~T * D * n_rows * n_features = 500 * 8 * 10^7 * 500 = ~2*10^13 ops (histogram-based) => ~1 hour on a 32-core machine. Retrain weekly + out-of-cycle on drift.

## Memory
- Raw storage: ~1 MB of raw data per user (activity history, texts) * 10^9 users = ~1 PB in cold storage (S3/HDFS).
- Online feature store: only the compact feature vector is needed - ~500 features * 4B = ~2-5 KB per user => 2-5 TB total. Fits in a sharded in-memory/SSD KV cluster; the hot subset (active users, ~10^8) is ~500 GB and can be cached in RAM.
- Graph: 10^11 edges * 16B = ~1.6 TB as an edge list, less in a compressed adjacency (CSR) format - distributed across the Spark cluster.
- Model: ~3 MB, negligible.

## Latency
The pipeline is asynchronous (Detection Service -> Queue -> model), so the user doesn't wait for the answer. The SLA is the time from the trigger to the decision, not the time of a single request.
- Feature fetch: 10-50 ms (dominates)
- Model scoring: ~5 us
- Cluster-level aggregation (score propagation over the linked cluster): ~10-20 ms
- Total processing: <100 ms per user, p99 < 200 ms
- End-to-end SLA from the trigger: p99 < 1-2 s for complain-driven checks (a high-priority queue), minutes are acceptable for the Sweep Detector Service. Under a backlog the queue wait dominates everything else - that's why the queue is prioritized.
- Feature freshness: graph features from the daily batch can be up to 24 h stale, so all time-critical signals (activity rate, complains, burst registration) must come from the streaming counters.