# Application
- positive/negative user feedback, news, reviews, etc.
- spam/fraud/rule violation detection
- tagging of feed (#News, #Sport, #IT, etc.)

# Offline Metrics
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- macro averaging for multiclass

- user feedback:
    - PR-AUC
    - Recall at fixed Precision

# Online Metrics
- user feedback: 
    - avg time between negative user feedback and support response (can be measured by time of feedback and time of first message in support chat relative to this feedback)
    - eventual user mark of support work (the faster support solves the problem - the higher the mark)
    - number of CMEs reports on incorrect model scoring (look at inference)
    - can also use user churn but it's noisy metric that depends on many things
    - guardrails: support and CME load; roll out with a shadow launch, then an A/B test
- fraud detection: 
    - number of user complains on fraud
    - money lost to fraud per 1k transactions or users, including chargebacks
    - share of blocks reversed on appeal, which is a direct signal of false blocking
- tagging:
    - Tag CTR (click-through rate): clicks on tags per tag impression
    - put in the complain menu "Incorrect tag" point and count the number of complains like this for users

# Data
Positive/negative feedback:
    We are interested only in feedback text and wether it's positive of negative, user and text metadata are not interesting
    
    Use CMEs to process texts and mark them as positive/negative feedback. 

    Also users can rate the feedback with stars or marks, those can be used for source for labels (1-2 stars = negative, 4-5 = positive, and 3-star feedback is excluded or sent to CMEs). But still that needs additional check with CMEs. The model targets feedback without stars (chats, free-text forms); stars are only weak training labels.

    Train: target-channel feedback (CME and LLM labels) plus starred reviews as lower-weight weak labels. Val and test: only CME-labeled feedback from target channels.

    Then we create a golden dataset and prompt an LLM for further marking. Validate the prompt and LLM settings on golden dataset.
    Examples of marking for prompt could be taken from golden dataset. LLM result should be randomly checked with CMEs.

# Train
Positive/negative feedback:
    - Baseline TF-IDF + logistic regression, then BERT-like + classification head
    - Train/val/test split, group by user to avoid data leakage (some of users leaves only positive or negative feedback)
    - Train with the natural ratio and class weights (downsample the majority only if imbalance is severe). Keep the real ratio on val and test. Calibrate and choose thresholds on val.
    - BCE loss

# Inference
Positive/negative feedback:
    - Every user feedback with metadata is read and put to queue
    - Model processes user feedbacks in batches and output their scores
    - Feedbacks with confident negative scores are sent to support immediately, feedback with non-confident scores - to CMEs who decide
    if it's negative feedback or not, later this marks can be used for additional training of model on hard negatives
    - Random scored feedbacks can be sent to CMEs too (source of hard positive/negatives for train + model monitoring)

