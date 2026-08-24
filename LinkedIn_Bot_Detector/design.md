# LinkedIn Bot Detector

## Business Task
Detect bot behaviour and block them to prevent their interaction with users. The business task is to increase user's engagement and loyalty to the platform -> increase profit from subscription and ads.

## FT
- detect bot-like behaviour 
- the detection trigger is a suspicious behaviour - a lot of complains on user, connection from the unusual location (e.g. different country), using VPN and proxies, a lot of actions for some period of time, mass DMs, job applies or connection requests etc.
- in case of high confidence block the corresponding account automatically
- in case of mid confidence inform the Operator about a potentially bot-like behaviour

## NFT
- 1 billion users
- DAU 100 millions
- Suppose that 1 of 10 users demonstrates suspicious behaviour every day RPS: 10^8 / (10 * 10^5) = 1000 RPS
- The ratio of False Positives should be less then 1 to 1000 users

## ML task
Binary classification task with class imbalance handling

## Offline metric
PR-AUC
Fix the Specificity (e.g. 0.999) and increase Recall for that specificity
Both of those metrics are sustainable to a huge class imbalance

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
    - share of neighbours that are blocked or belong to the same linked cluster
    - PageRank / k-core position - real accounts are embedded in dense professional communities, farm accounts sit on the periphery
    - coordination features: number of other accounts that act on the same targets/posts within a small time window, similarity of activity time distribution to other accounts of the cluster
    - node2vec/GraphSAGE embedding of the user as a dense feature

Data example: user features + blocking history + last 50-100 activities + list of connections and how many of them were blocked for bot-like behaviour + linkage/graph features + additional features

## Model
Two approaches:
- Baseline: GBTD, then can switch to BERT-like transformer, which is good for sequence processing.
- Graph part: start with hand-crafted linkage/graph aggregates as features for GBTD (cheap, recomputed in batch), then can switch to GNN (Graph Neural Network, e.g. GraphSAGE) over the connection/interaction graph and feed its embedding into the main model.
- Cluster-level decision: score the linked account cluster as a whole, not only the single account - if a large share of the cluster is confidently bot-like, raise the score of the remaining accounts (a single account is easy to disguise, a farm of 10k is not).

Fine-tune the threshold based on offline metric

## Train
Train/val/split based on user_id and bot campaings (bots from the same bot campaing must belong to either train or val datasets), optimize BCE.

How to define that user is bot? He/she was blocked for bot-like behaviour and didn't appeal his/her blocking during some time (e.g. one month) or appelation was failed (e.g. failed to KYC) and no additional attempts were made in one month. This bot accounts must be additionally verified by CME to prevent false positives.

## Inference
- Detection Service detect the suspicious user behaviour
- It collects user data and put them into Queue
- Then model takes the user data from the Queue
- Model returns the score:
    - If it's high - user is blocked automatically (can appeal this later)
    - If it's high but not enough - report to a human operator - let him/her decide

## A/B tests + montoring
At first check offline metrics
Then try the model on 1-5% of users, measure:
    - Avg time that user spends in application
    - User activity
    - Number of bought subscriptions

Control number of complains, appeals and churn rate: if something of that is growing rapidly - stop the experiment and find out why this happened.
If all okay - increase the number of users

## Fallback
In case of model failing - just check only users that are complained by and do the check manually using Operators.

## Compute

## Memory

## Latency