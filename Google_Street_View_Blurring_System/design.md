# Blurring System for Google Street View

## Business task

Protect the personal data + minimize the cost of manual labor for car plate blurring.

## Functional requirements (FT)

- Blur human faces and car plates on GSV
- Process any types of car plates
- Tackle with biased racial distribution of human faces
- User can complain about the face/car plate which was not blurred
- Some pictures can randomly be sent to operator for manual verification of blur correctness

## Non-functional requirements (NFT)

- How many photos in total? 100M
- How many new photos per day? Suppose that million per day → RPS ~ 10 → load on system is low
- Is system latency critical? Don't think so, users can use previous photos until new photos are processed, not latency critical for the blur-production pipeline; the map-lookup/serving path is out of scope for this design

## ML task

- **Semantic segmentation** (because we need to detect and mask exactly car plate or exactly human face and don't touch any other objects nearby which can decrease informativity of photo + if e.g. multiple faces are close to each other — we don't need to separate them from each other, just blur them together)
- Do we have labeled dataset for this task? 1 million of labeled photos

## Data

- Photo
- Dataset: (object type (face, plate), photo URL, array of masked pixel coordinates)
- Metadata: timestamp, coordinates, camera orientation (X, Y, Z angles)

Need to tackle with limited diversity of dataset: split the original dataset into clusters by skin color and use stratified sampling while training.

## Model

- Preprocess pipeline (resize + convert to universal color scheme (RGB/CMYK))
- Data augmentation (crop, scale, mirror, change colors/brightness/contrast, cutmix, noise)
- Semantic segmentation architecture (U-Net, FCN, DeepLab)

## Loss function

Dice loss + BCE loss.

Dice computes a ratio of overlap mask and real object to total mass of both — it is robust to class imbalance. But Dice is unstable when GT mask is very small or empty (that's why epsilon is added) or saturates when overlap is roughly correct. That's why Dice is combined with BCE.

```
Dice = (2 * sum(p_i * g_i) + epsilon) / (sum(p_i^2) + sum(g_i^2) + epsilon)

Dice loss  = 1 - Dice
BCE loss   = - sum(g_i * log(p_i) + (1 - g_i) * log(1 - p_i))
Total loss = alpha * Dice loss + beta * BCE loss
```

- `p_i` — model prediction for i-th pixel
- `g_i` — 1 if i-th pixel belongs to class, 0 otherwise
- `epsilon` ~ 1e-8

## Offline metrics

- IOU (intersection over union)
- mAP (mean Average Precision)
- Prepare golden dataset for evaluation of face blurring for different races and lighting conditions
- Threshold for TP/FP should be biased toward higher recall (because FP blur is a small inconvenience but FN blur is a personal data exposure) → threshold should be lower

## Online metrics

- (Number of user complaints about not blurred human faces / car plates for the last day/week/month) / (Total number of users for the last day/week/month)
- (Amount of manual blurrings for the last day/week/month) / (Total amount of processed photos for the last day/week/month)

## Inference

- Train dataset keeps increasing:
    - Audit service randomly grabs images (the probability can be adjusted by model confidence — the lower model confidence is, the higher probability)
    - Audit service asks operator to check images (whether predicted masks are correct)
    - User complains about not blurred human faces / car plates
    - Operator fixes them and adds new images with correct masks to train dataset
- Rollout model with 1% of data → collect metrics → gradually increase percentage
- Retrain model when at least 10% of new data are added or performance (measured by Offline and Online metrics) of new model degrades comparing to previous model

## A/B tests

- Randomization unit: **image**, not user — output isn't personalized, every viewer sees the same blur, so there's no cross-request correlation to worry about like in a ranking system
- Still assign by capture batch/geographic tile rather than pure per-image random: photos from the same drive are correlated (same lighting, camera, plate/face style), so naive per-image randomization can accidentally concentrate one region's edge cases into a single arm
- Organic complaints are too rare and too slow to reach significance on their own (long-tail event, reported days/weeks after exposure) — the audit service's forced random sampling is the actual fast-turnaround readout, not the passive complaint stream
- Shadow mode before any real exposure: run the candidate model on live incoming photos in parallel, log where its predicted masks disagree with the current model's, route disagreements to an operator for adjudication — estimates the real precision/recall delta with **zero risk**, since the candidate's output is never published
- Rollout ladder, each step gated on the previous. Unlike a typical UX A/B this is not reversible once a photo ships unblurred, so start far more conservative:
    1. Offline: IOU/mAP on golden dataset
    2. Shadow mode on live traffic (no publish)
    3. Canary: 1% of newly captured photos, elevated audit-sampling rate on that 1% to get a fast read on FN/FP instead of waiting on complaints
    4. Ramp 1% → 10% → 50% → 100%, guardrails re-checked at each step, automatic rollback if breached
- Guardrails: manual-blurring rate must not increase (model pushing more images into human fallback), predicted-blur-area distribution must not shift heavily (a collapse to "blur everything" trivially wins recall while destroying informativity — the FN-biased threshold already accepts some over-blur, but a distribution shift flags a degenerate model)
- Long-running holdback: small % of traffic kept on the previous model indefinitely, to catch slow drift a 2-4 week A/B window can't see — new car models, new plate formats/fonts, seasonal clothing changes affecting face detection

## Monitoring

- Model behavior (no ground truth needed, real-time): predicted-blur-area per photo and detections-per-photo, tracked as a distribution — a drop toward zero across a slice is the dangerous case, since it produces no errors, just silently unblurred photos
- Per-slice coverage, not just aggregate: blur rate broken out by region/camera hardware/capture batch — an aggregate blur rate can look perfectly normal while one region or one new camera model silently gets near-zero detections (e.g. a feed integration bug skips preprocessing for that source), and it would take a while for complaints from that specific region to surface it
- Input drift: new camera hardware (different color profile/resolution), new geographic rollout (unfamiliar plate formats), seasonal lighting shift — same class of preprocessing-skew risk as any vision pipeline
- Pipeline/infra: capture-to-blur processing lag (queue backlog), preprocessing error/reject rate (corrupted or unsupported images), GPU utilization of the segmentation fleet
- Quality proxy with ground truth: audit-service agreement rate between model prediction and operator correction — same signal used for the retrain trigger, so it should be dashboarded and alerted on, not just checked at retrain time
- Business/compliance (see also GDPR section above): user complaint rate and manual-blurring rate trends — a spike here is an **incident**, not a metric to note on a dashboard, since by the time it's visible the underlying privacy exposure already shipped to production
- Deployed model version vs. version used to build the current golden/audit baseline must match — same hard-alert requirement as any versioned model, a silent mismatch invalidates every metric above it

## General Data Protection Regulation (GDPR)

- Some images need to be stored unblurred (for model retraining), but only for internal purpose
- Most original images have TTL (e.g. 30-90 days) because of audit purpose
- In case of complaint from user — TTL for original image can be frozen until complaint is resolved
- Very few roles have access to unblurred images, every access must be logged
- Users have right to request their personal data (including any unblurred images) to be deleted: they just need to select image on which they (presumably) appear and send request, their data will be indexed by geolocation and if they appear in any nearby street view images they all will be deleted too

