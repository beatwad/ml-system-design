# Application
    - doc processing - pulling names, dates, amounts, etc.
    - PII masking - hiding names, phones, card numbers, passwords
    - obtaining of search fields from query (e.g. brand, price, delivery region)
    - support chatbots - extraction of order number, products, cities

# Offline Metrics
    Most tokens are 0, so classes are imbalanced.
    - precision
    - recall
    - F1
    Use entity-level with strict matching and macro averaging with exlcuded plain words.

# Online Metrics
    Doc processing - user sends a document file or photo, system pulls entities from it and fills all necessary fields. When system pulls all necessary data, it fills fields with it and asks user to check if it's OK, if user checks ok, than move next, else:
        - count the number of cases when user says it's not okay
        - count the number of entities that were not found
        - track the fields that user changed for further training
        - guardrail: p95-p99 latency, audited errors in accepted fields

# Data
    - use CMEs to mark some number of texts using pre-defined BIO tags: plain text is O, first token of name is B-NAME (begin), second, third, etc. token of name I-NAME (internal), first token of DATE is B_DATE, etc.;
    - use field-specific labels (DUE_DATE, INVOICE_DATE);
    - use this Golden Datased to create a prompt for an LLM with examples;
    - then use the LLM for further marking, add guardrails (e.g. LLM returns entities as JSON ({"text": "John Smith", "type": "NAME"}), find them in the source text and reject the example if one isn't found);
    - control quality using CMEs.

# Train
    - baseline: LLM that was used for mark;
    - model: use BERT-like. Add matrix with shape (D, C) as head, where C = 2 × entity types + 1 (2 types of entity tokens + O);
    - so the model output now is [B, T, C] instead of [B, T, D];
    - how to match words and tokens: the first subtoken gets the word's label, and all other subtokens and special tokens get -100, loss function ignores them;
    - train/val/test split, val and test consists only of texts that were processed with CMEs; 
    - group by document template or issuer;
    - train on OCR output, not clean text;
    - split long documents into overlapping windows;
    - loss is CE.

# Inference
    - extract data from document using OCR put them to queue;
    - split long documents into overlapping windows;
    - data are sent to a model, argmax on the model's output at each word's first subtoken, then merge BIO tags into entities;
    - validate format of each found entity;
    - if score for some enitities is low - mark them for user;
    - if get multiple entities for one field, select the one in which model is sure, but warn user that there are more than one candidate;
    - fields are then filled with recognized entities;
    - if some fields weren't found - ask the user to fill it him/herself and log this for further train.