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
- auctions run in real time, so the latency matters: 50-100 ms per request

# ML task

It's a ranking task, i.e. we have pool of available ads that we can show to user, our task is to rank them and show the most relevant. But also we must predict probability of the user click, to correctly predict revenue. That means that our model must be calibrated.

# Offline metric

We have ranking task, but we also must predict a probability of click. Relevance is binary (clicked or not), multiple ads can be clicked + task is imbalanced (user clicks only small fraction of shown ad)

So offline metric:
- Expected Calibration Error: bin predictions by predicted probability, then take the weighted average of the gap between mean predicted probability and observed positive rate in each bin:
`ECE = sum (n_b/N * |obs(b) - pred(b)|)`
- ROC-AUC, mAP doesn't work good because we don't have labels for a candidates with low score - that wasn't shown at all, only for candidates with score that is high enough. So we need a global pairwise metric that shows 
- Normalized Cross Entropy:
`NE = BCE / -(p' * log(p') + (1 - p') * log(1 - p'))`

Where:
- `b` - bin in sample of objects, sorted by model confidence
- `n_b` - number of objects in bin
- `N` - total number of objects
- `pred(b)` - mean predicted probability of click in bin
- `obs(b)` - mean observed probability of click in bin
-  `p'` - the empirical posititve rate (avg CTR) of the sample

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
- activity history (posts, views, likes, shares, comments, time)
- ad intercation history (what ad was shown (impression - ≥50% of the ad visible for ≥1 s), what was clicked (a click within N minutes of the impression, don't consider accidental clicks (bounce under ~2 sec)), what was hided or scrolled, time, device, ad position, what was already shown in this session, etc.)
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

# Model

For that task BERT-like cross-encoder is suitable, but it's too heavy. So we use two-staged system:
- lightweight bi-encoder model or GBDT as retriever
- cross-encoder BERT-like model as reranker

# Loss function

- InfoNCE for retriever
- BCE for reranker


