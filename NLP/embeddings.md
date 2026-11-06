# Application
    - RAG
    - Recommender Systems
    - Search of similar item/video/article

# Offline Metrics
    - RAG: 
        - Recall@K
        - NDCG@K
        - MRR@K
        - MAP@K

# Online Metrics
    - RAG:
        - number of search refusals and reformulations (user stopped the search without result or tried to change the query)
        - citation CTR
        - thumbs up/down rate on the answer
        - avg number of messages for successful search (LLM should return the correct answer with first message)
        - deflection rate - measured reduction in support tickets/SME
        - DAU/WAU/MAU
        - guardrails: number of user requests to support, p95 latency

# Data
    - use CMEs to create a Golden Dataset with pairs of query and answer with proper citations from appropriate documents
    - then use LLM to create more query-answer pairs, can create answer from ready-made question or create question for some document
    - split documents on chunks, for each query-answer pair find out which chunks are used in answer
    - create positive pairs question-chunk


# Train
    - pretrain baseline: E5 or BGE
    - use Bi-Encoder model
    - train/val/test split, group by documents, val and test are CME-labeled only
    - for every batch take N queries and N chunks, build a query-chunk matrix with positive labels on diagonal, take care that every query in batch has only one relevant chunk and vice versa
    - add hard negatives to batch from another models (e.g. find the most closest irrelevant chunks for the query using BM25)
    - use Info-NCE loss

# Inference
    - use document encoder to encode every chunk, than put that chunks to vector database using ANN index (e.g. HNSW)
    - encode query with query encoder of the model
    - use query embedding to find top K relevant chunks
    - send them to the next stage (e.g. combine them with the results of lexical model like BM25 using RRF and send them to reranker)
    - in case of model retraining need to re-embed all chunks in the database