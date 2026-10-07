# Ads CTR Prediction for a Social Media Platform

You work at a large social network with a feed similar to Facebook or Instagram. The company makes most of its revenue from ads shown in users' feeds.

When a user opens or scrolls the feed, there are ad slots to fill. Advertisers set up campaigns with targeting, a budget and a bid, and mostly pay per click (CPC, Cost Per Click). The platform runs an auction to decide which ad goes in each slot, and it ranks candidates by expected revenue: bid × P(click).

The current click predictor is a logistic regression with hand-crafted features. It has stopped improving, and advertisers complain that their spend is going to low-quality impressions.

**Your task is to design an ML system that predicts the probability a given user clicks a given ad in a given context, so the auction can rank and price ads.**

A few points the interviewer cares about but won't state up front:

- The predicted probability is multiplied by money, so it has to be **calibrated**, not just ranked correctly.
- The model needs to stay accurate as ads, campaigns and user interests change.
- Ads should not hurt the experience of the people using the feed.

# Business task

Show more relevant Ad to user -> user clicks more often -> social network gets more revenue. But don't let clickbait advert to conquer the user's feed.

# FT

- predict which ad to show user to maximize bid × P(click)
- ad must be diverse and do not repeat, so the user don't get tired of repeating ads
- don't show user an advert which he/she marked as undesireble
- consider age and region restrictions (e.g. don't show 18+ ad to children or don't show beer ad to Saudi users)

# NFT

- 1 billion users total
- DAU 100M 
- each user sees 10 ads per day
- 10^9 ads per day total -> ~10000 RPS
- peak load ~3x -> ~35000 RPS
- ~500 candidates scored per request
- auctions run in real time, so the latency matters: 50-100 ms per request

# ML task

It's a binary classification task: predict probability of the user click (pCTR) for each candidate ad. The auction then ranks ads by bid × pCTR, so pCTR must be calibrated to correctly predict revenue.

# Offline metric

We have binary classification task, but predicted probability of click must be calibrated. Relevance is binary (clicked or not), multiple ads can be clicked + task is imbalanced (user clicks only small fraction of shown ad)

So offline metric:
- Expected Calibration Error: bin predictions by predicted probability, then take the weighted average of the gap between mean predicted probability and observed positive rate in each bin:
`ECE = sum (n_b/N * |obs(b) - pred(b)|)`
- ROC-AUC (not mAP, because we don't have labels for a candidates with low score - that wasn't shown at all, only for candidates with score that is high enough. So we need a global pairwise metric that shows how well pCTR separates clicked impressions from non-clicked ones)
- Normalized Cross Entropy:
`NE = BCE / -(p_avg * log(p_avg) + (1 - p_avg) * log(1 - p_avg))`

Where:
- `b` - bin in sample of objects, sorted by model confidence
- `n_b` - number of objects in bin
- `N` - total number of objects
- `pred(b)` - mean predicted probability of click in bin
- `obs(b)` - mean observed probability of click in bin
-  `p_avg` - the empirical posititve rate (avg CTR) of the sample

For retriever we use Recall@k metric.

# Online metric

- RPM (Revenue Per Mille, i.e per 1000 impressions)
- CVR (Conversion Rate after the click)
- ROI (Return of Investment)
- ad complain/hide ratio
- Avg user time spent (can fall drastically in case when users see a lot of irrelevant ads)
- online predicted/observed click ratio

# Data

User data:
- user id
- age
- gender
- country
- city/region 
- language
- interests list (e.g. tags like #sciense, #history, #music)
- friend list
- activity history (posts, views, likes, shares, comments,)
- ad intercation history (what ad was shown (impression - ≥50% of the ad visible for ≥1 s), what was hided or scrolled, what was clicked, etc.)
- user x category CTR counters (1h / 1d / 7d)

Ad data:
- ad id
- ad text
- ad image or video
- advertised item id
- item category
- advertiser id
- campaign id
- user intercation history (e.g. historical CTR, to what user category was shown and when)

Request context (current request):
- time of day, day of week
- device, OS, surface (feed, stories, etc.)
- ad slot position (it's set to a constant at inference)
- ads already shown in this session

Target: a click within N minutes of the impression, don't consider accidental clicks (bounce under ~2 sec), so the training pipeline has to wait for N minutes before labeling an impression as negative to prevent biased pCTR.

# Model

~500 candidates x ~35000 RPS within 50-100 ms, and features are mostly IDs, counters and context (not text), so we use two-staged system:
- lightweight bi-encoder model or GBDT as retriever
- DCN-v2 as reranker. Input: embedding tables for IDs (user, ad, advertiser, campaign, category), dense counters and request context, precomputed ad text/image embeddings, Deep Interest Network (DIN) attention over the user's recent ad interactions

# Loss function

- InfoNCE or LambdaMART (depends on model) for retriever
- BCE for reranker

Negative downsampling: ~10^9 impressions/day with ~1% CTR, so we keep only a fraction `w` of negatives. It inflates predicted CTR, so we recalibrate the output:
`p = p' / (p' + (1 - p') / w)`, where `p'` - prediction of the model trained on downsampled data

# Train

Time series split, gap N minutes between train and validation. 
- at first train retriever: for each user select ads that were clicked, add ads that were shown but not clicked + ads that *never shown* (they will be negatives with a high probability too) and train using InfoNCE with in-batch negatives + logQ correction, plus explicit shown-not-clicked negatives. logQ correction - subtract the log of each ad's sampling probability `Q(a)` from the logit: `s_corrected(u, a) = s(u, a) − log Q(a)`. Popular ads often appear in batch as negatives, so the model learns that popular -> low score; the size of that bias is -log Q(a), so the correction cancels it. Applied to logits of all in-batch candidates (positive included), not to explicit shown-not-clicked negatives. Training only, serving uses raw `s(u, a)`.
- then train reranker, pointwise with BCE on all logged impressions, with uniform negative downsampling
- then distill reranker into retriever using soft labels + add hard negatives and positives

Retrain: do a daily full retrain and hourly incremental updates, cause ads and campaigns change fast.

# Inference

What triggers the ad recommendation? User launches the app / opens the site / swipes feed / any other action that leads to ad show.

We get user embedding, pre-filter available ads by age, region restrictions, frequency (was shown to that user less than M minutes before), campaign has remaining budget (pacing), etc.

We get ad embedding, we load history of it's interactions with various user categories (e.g CTR counters). CTR is calculated using formula `(clicks + α·category_CTR) / (impressions + α)`, that means that if ad wasn't shown, clicks = impressions = 0 and we use avg CTR of the corresponding ad category.

New ads rely on content embeddings (text/image) because their ID embedding is untrained; a small exploration share of traffic is reserved for them.

For the current user embedding we search for top K (e.g. 500) most similar ads in filtered ANN index (pre-filter conditions are applied as attribute filters inside the search), then rerank them, then apply calibration (downsampling correction, **isotonic scaling** - a post-hoc calibration using a separate small function that maps raw scores to calibrated probabilities), compute bid × pCTR per candidate, pick the winners (no same advertiser in adjacent slots), set the price.

If isotonic scaling is used:
- apply it after downsampling correction
- refit it daily or hourly
- optionally fit it per segment (country, surface, new vs. established ads, etc.)

Log the exact feature values used together with the impression, and build training data from those logs for further retraining.

# A/B testing

## Primary
- Revenue per Mille

## Guardrails
- User time spent
- Predicted/Observed click ratio
- Ad complain/hide ratio

## Secondary
- Advertiser CVR/ROI
- Avg Impression duration (normalized per ad format)

- Randomize by user.
- Use budget-split testing: advertiser budgets are shared, so one arm can spend the other's budget and inflate its own RPM.
- Run ≥1–2 weeks for weekly seasonality and novelty effects.
- Keep a long-term holdout to measure the effect of ad load on retention.

# Monitoring
- Predicted/observed click ratio
- RPM
- CTR
- Ad complain/hide ratio
- Avg session length
- Avg user time spent
- Feature/Target/Concept drift
- Model latency (median, p99)

In case of calibration drift - refit isotonic calibration
In any other case case when some of monitoring parameter deteriorates signifincantly - send notification to Operator

# Fallback
- Reranker fails - fall back to the previous reranker version. If none is available, use the smoothed historical CTR counters as pCTR; they're calibrated by construction.
- Retriever fails - use precomputed cache of top ads per user segment, ranked by smoothed historical CTR, refreshed hourly
- New model/features shows significantly worse performance - return previous version of model/feature
- Filter service - show only ad without age/region restrictions

# Compute

Assumptions: ~10^6 active ads, peak 3.5*10^4 RPS, K = 500 candidates, A100 ~ 3.12*10^14 FLOP/s fp16 with MFU ~0.3.

Online:
- Reranker (DCN-v2 + DIN): ~40 MFLOP per candidate -> 500 * 4*10^7 = 2*10^10 FLOP per request -> 3.5*10^4 * 2*10^10 = 7*10^14 FLOP/s at peak -> ~7 GPU at 100% util, ~12 GPU at 60% util, x2-3 for regions/redundancy -> ~30 GPU. On CPU it would be ~10-20k cores, so GPU is cheaper.
- Retriever: user tower is a small MLP, negligible. Filtered ANN (HNSW) over 10^6 ads ~1-2 ms per query on 1 core -> 3.5*10^4 * 2 ms = 70 cores -> ~100-150 CPU cores.
- Feature store: ~100 keys per request (user features, user x category counters, DIN history) -> 3.5*10^6 lookups/s -> sharded Redis. Ad-side features (10^6 ads * ~1 KB = 1 GB) are cached locally in every reranker node and refreshed every minute, so 500 candidates don't need 500 remote lookups.

Offline:
- Training data: 10^9 impressions/day, keep w = 0.1 of negatives -> ~1.1*10^8 rows/day.
- Full retrain on 30 days: 3.3*10^9 rows * 3 * 4*10^7 FLOP = 4*10^17 FLOP -> ~1-2 GPU-hours of pure compute. In practice bound by embedding lookups and data I/O -> ~2-4 hours on 8-16 GPU, daily.
- Hourly incremental update: ~5*10^6 rows -> minutes.
- Ad embedder runs only on ad creation/update -> negligible.

# Latency

Budget: p99 < 100 ms.
- User features + user embedding: 5-10 ms (in parallel)
- Filtered ANN top-500: 5-10 ms
- Ad features from local cache: ~1 ms
- Reranker, 500 candidates in one GPU batch: 5-15 ms (+ few ms of dynamic batching queue)
- Calibration + auction + business rules: ~1-2 ms
- Logging: async, 0 ms
- Network/serialization: 10-20 ms
- Total: ~40-60 ms median, p99 < 100 ms. If over budget: lower K (500 -> 200), cache user embedding per session.

# Memory

- ID embedding tables: users active in last 30 days ~3*10^8 * 64 * 2B (fp16) = ~40 GB, ads 10^7 * 64 * 2B = ~1.3 GB -> fit on one 80 GB GPU or sharded across 2-4 GPUs. Dense part (cross layers + MLP + DIN) ~10^7 params = ~20-40 MB, replicated.
- Vector DB: 10^6 ads * 128 * 4B = 0.5 GB + HNSW graph -> ~1 GB, replicated in every retriever node.
- Feature store: ~200 user features + user x category counters (100 categories * 3 windows * 2 counts * 4B = 2.4 KB) + DIN history (100 recent ad interactions * ~16B = 1.6 KB) -> ~5 KB per user * 10^9 = ~5 TB on SSD KV. Hot subset (10^8 DAU) ~500 GB in RAM.
- Logs: 10^9 impressions/day * ~2 KB of logged features = ~2 TB/day -> ~180 TB for 90 days retention in cold storage (S3/HDFS).

