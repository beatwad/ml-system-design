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


