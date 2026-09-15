# Item Sales Anomaly Detection System

A huge retailer wants to build a system that detects the situation when item sales fall drastically comparing to expected sales. Currently this task is solved by analytics who manually process sales plots and look for anomalies. Retailer wants the system to automatically detect the anomalies and notify analytics who then investigates the reason of anomaly manually.

## Business task
We lose $10k for each missed anomaly. Avg anomaly rate is 10 anomalies per month -> $100k of potential loss. Couple of Data Scientists who build and support the system is $100k per year. So it makes sense. 

## FT

- monitor sales for pairs item-shop
- monitor sales of items for all shops
- system must send concrete notifications (what, when, where, what was before)
- project manager should get a regular month reports about system efficiency

## NFT

- 1000 shops
- 10k different items
- 5k invoices per hour
- minimize number of false alarams
- mimimize time of anomaly detection
- maximize number of detected anomalies

## ML task

Anomaly detection. We predict sales of pair item-shop or item for all shops for the next week/day/hour and if expected << observed - it's anomaly.

## Offline metrics

Use quantile regression with q <= 0.01, i.e. penalize model for a huge overprediction.

For each item / item group we set the treshold and if difference between predicted target and observed target is bigger than this threshold - it's anomaly.

Offline metrics:
- Recall (need to catch as many anomalies as possible)
- Precision (minimize false alarams)
- F-beta (beta > 1 because we care about Recall more)
- PR-AUC

## Online metrics

- Real anomalies / All notifications
- Number of false alarms
- Number of missed anomalies (analitics team can work in parallel and detect anomalies manually, than we count the number of anomalies that we detected by them and was not detected by the system)
- Time between anomaly appearing and detection (can be derived from history)

## Data

We have 2 years of data -> ~100M invoices. Can group them by total sales per shop/shops per hour/day/week/month. 

- shop id
- item id
- timestamp
- quantity
- price
- shop coords
- previous sales (min/max/avg sales hour/day/week/month ago)
- sales of that item in N closest shops
- sales of similar items in that shop (maybe in closest shops)
- don't have customer info
- can get additional data, e.g. weather forcast for this place and time
- out of stock flag (to filter the situation when sales suddenly fall because of out of stock situation)
- target must be normalized to avoid price difference: `|current - previous| / previous` or `|current - previous| / ((current + previous) / 2)`

Need to have a feature storage which updates every hour/6 hours/day and then prediction model is run.

## Model

We want to minimize time of anomaly detection - need relatively simple model:
- Linear Regression
- SVM Regressor
- RandomForest Regressor
- Gradient Boosting

Model is run for ~10M item-shop pairs and 10k items for all shops -> ~10M objects as frequently as possible, let it be 1 hour. All of these models above are light weight enough, 10M predictions/hour is okay for them. Problem mostly in feature preparation.

## Train

Prepare features and target, use TSS.

How many item-shop pairs if we predict daily for 2 years? 80B - quite huge.

How many total invoices? 60M -> the majority of items are sold rarely -> except different threshold we should also predict more frequently for some items and less frequently for others. This should be determined during train.

## Inference

Use something like cron job, for each item category we fire with some period (1 hour, 2 hours, 8 hours, daily, weekly, etc., controled by Scheduler), prepare features, send them to model, make prediction, compare with item-specific threshold, notify Analysts if necessary.

Also have Monitoring Service which detects feature/target/concept drift and retrains the model + Data Collection Service to collect information from Analysts about anomalies that were not detected and add them to train data. Thresholds and periods for each item/group of items can be set by Setting Service.

## Monitoring

- feature/target/concept drift
- number of notifications for some period of time
- number of missed anomalies for some period of time

## AB-test

Run the model in parallel with analysts for e.g. a month.

Primary metrics:
- recall, must not be significantly worth than analysts recall
- time between data appearance and system detection, must be significantly better than analysts detection time

Secondary metrics:
- precision, not so crucial as recall, but still should be controlled to prevent analytics flooding with system notifications in future

Control metrics:
Use monitoring metrics for AB-test control.

## Fallback

If new item appears in stock: just wait for some time until we collect enough data about this item sales in that shop and retrain model on them + can use information about this item's sales from nearby shops (if that item exists for some time in the shops nearby). Same for the new shop - wait for data to collect + use information from nearby shops.

If feature storage of model is failed - notify analysts and ML engineers about the situation.

If number of false notifications or missed anomalies rise dramatically (e.g. +100%) - notify ML Engineers, switch back to previous version of model if neccessary or retrain new model.

## Data

60M of invoices for 2 years + aggregates + features, suppose 1 kB per invoice -> ~60 GB of data + 30 GB of new data each year - one 1 TB SSD disk is enough for the next 5 years.

## Compute

Models are light, compute of features can take time. Suppose it's 20M of pairs item-shop per 24 hours, each pair has 10kB of features to genearate -> 200 GB of data to process every 24 hours. Prepare 2*10^7 pairs / 10^5 secs is 200 pairs per second. Doesn't look computationally intensive. 

## Latency

Model takes microseconds. Million of rows for one GBDT per 1 CPU core is ~30 secs, multiple CPU cores decrease this time to seconds. Feature preparation will take tens of seconds -> the latency of whole system is less than a minute even on weak hardware.