# Application
- post-processing ASR (Automatic Speech Recognition) output

# Offline Metrics
- majority of tokens have class O, so this is imbalanced multiclass classification:
    - Precision
    - Recall
    - F1
    - PR-AUC
- use word-level with macro averaging, report per-head and per-class exclude plain words for register head and words without punctuation marks for punctuation head.

# Online Metrics
- user edit rate (punctuation and register) per 1000 words, compare in A/B tests;
- guardrails: p95 latency

# Data
ASR:
    - data are almost free: just take well punctuated text, bring it to low register and remove all punctuation marks;
    - simulate noise from ASR like random words, sound transcription, etc., align labels accordingly;
    - treat words like 3.14, one-two, URLs, etc. separately to avoid false positives.

# Model
ASR:
    - baseline: LLM with prompt;
    - predict on word level, e.g. select only first subtoken of word and predict only for it, second and next tokens are marked as -100 and don't participate in train and loss calculation;
    - split the text on random chunks before sending to a model;
    - model is BERT-like with two heads:
        - first head predicts punctuation marks after word (.,!?) or punctuation absense (O - majority of words)
        - second head predicts word register (lower, Capital, UPPER)
    - train/val/test split, group by text or text corpus;
    - val and test are build from the real ASR with human added register and punctuation;
    - tune threshold to maximize offline metrics, leave words and punctuation with low score untouched;
    - loss is punct_loss + lambda * case_loss, both are CEs, lambda is 1 by default;
    - handle class imbalance with class weights or focal loss;
    - when model is in production, can use manual CMEs check and user edits to extract hard examples (positives with low score and negatives with high score) for additional training.

# Inference
ASR:
    - get ASR output, split long text outputs into overlapping windows, if prediction belongs to two windows, select prediction that lies closer to corresponding window center;
    - preprocess the text and send it to the model;
    - for each word model outputs two predictions: register and punctuation, according to models' prediction transform the word and/or add punctuation mark after it;
    - postprocess text, e.g. if it's text beginning or previous punctuation mark is ./?/!, that means that next word must have capital register (if it's not already UPPER of course);
    - apply the dictionary last for special words like McDonalds or iPhone;
    - send random predictions for manual check.