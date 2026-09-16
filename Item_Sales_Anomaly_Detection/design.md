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
- minimize number of false alarams (not more than 5 per day)
- mimimize time of anomaly detection
- maximize number of detected anomalies

## ML task

Anomaly detection. We predict sales of pair item-shop or item for all shops for the next week/day/hour and if observed << expected  - it's anomaly.

## Offline metrics

For each item / item group we set the treshold and if difference between predicted target and observed target is bigger than this threshold - it's anomaly. More details about threshold in Model section.

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

We have 2 years of data -> ~60M invoices. Can group them by total sales per shop/shops per hour/day/week/month. Also take into account that not every item is being sold in every shop, so the total number of item-shop pairs << shop_num * item_num

- shop id
- item id
- timestamp + time features (season, weekday, holidays, etc)
- quantity
- price and price change
- promo, sales, price offs, etc.
- shop coords
- previous sales (min/max/avg sales hour/day/week/month ago)
- sales of that item in N closest shops
- sales of similar items in that shop (maybe in closest shops)
- don't have customer info
- additional data, e.g. weather forcast for this place and time
- out of stock flag (to filter the situation when sales suddenly fall because of out of stock situation) - we don't consider OOS as anomaly and won't use our model to detect it
- shop is not working flag (working hours, renovation, permamently closed)
- item was delisted from shop's product range
- target: item-shop sales for the last hour/day/week/month (it depends on item and shop)

Need to have a feature storage which updates every hour/6 hours/day and then prediction model is run.

## Model

We want to minimize time of anomaly detection - need relatively simple model:
- Linear Regression
- RandomForest Regressor
- Gradient Boosting

We use GBDT with Poisson or Tweedie objective to predict the number of sales for every item-shop pair, because these objectives use a log link, so the prediction is always positive and multiplicative effects like seasonality and promos come naturally.

We have 5000 invoices per hour, so fo some item-shop pairs we can predict rarely. Suppose we must predict for 100k pairs per hour. GBDTs like LightGBM or CatBoost are lightweight enough, 100K predictions/hour is okay for them and features will be prepared fast enough too.

After we predict the sales next hour/day/week/etc., we use this prediction as lambda in e.g. Poisson distribution and predict the probability to get the same or less sales that we observe (i.e. cumulative probability of the left tail of distribution). And if it's less than a threshold (alpha) for that item or item-shop pair - we send an alert. alpha must be derived from false alert budget (not more than 5 false alerts per day -> not more than 1 false alert per 3 hours).

## Train

Prepare features and target, use TSS. Use SMEs to show which of sales drops are anomalies and which are not. 

Also consider adding of syntetic sales falls cause their historical number is ~240 and this is a very small value for 60M rows dataset - this will help us to make Offline metrics not so noisy.

The majority of items are sold rarely -> except different threshold we should also predict more frequently for some items and less frequently for others. This should be determined during train.

Features, that contain anomaly sale behaviour, must be excluded from the train dataset - model must not treat them as kind of normal behaviour. E.g. we can replace them with mean of previous and next sales (if both of them ok).

## Inference

Use something like cron job, for each item category we fire with some period (1 hour, 2 hours, 8 hours, daily, weekly, etc., controled by Scheduler), prepare features, send them to model, make prediction, compare with item-specific threshold, notify Analysts if necessary.

Also have Monitoring Service which detects feature/target/concept drift and send notifications to ML Engineers in that case. 

Also periodically (e.g. once a week) retrain the model. Use Data Collection Service to collect information from Analysts about anomalies that were not detected and add them to train data. 

Thresholds and periods for each item/group of items can be set by Setting Service.

Even when model will be put production, analytics should conduct random manual anomaly checks from time to time for sales that model considers as not anomal.

## Monitoring

- feature/target/concept drift
- number of notifications for some period of time
- number of missed anomalies for some period of time

## AB-test

Run the model in parallel with analysts for e.g. a month + add a synthetic anomalies to increase the amount of data (10 anomalies per month is not enough for reliable test).

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

Models are light, compute of features can take time. Suppose it's 100k of pairs item-shop every hour, each pair has 1kB of features to genearate -> 100 MB of data to process every hour. Even weak hardware can handle it.

## Compute and Latency

 Model takes microseconds. Million of rows for one GBDT per 1 CPU core is ~30 secs, 100k -> 3 secs, multiple CPU cores decrease this time to less than a second. Feature preparation will take tens of seconds for million obects -> couple of seconds for 100k -> the latency of the whole system is less than a 10-20 secs even on weak hardware.