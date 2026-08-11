---
name: mlsd-review
description: Use this skill whenever the user is preparing for an ML System Design interview and shares a solution to a design task — as an image/screenshot of a diagram/scheme, a text writeup, or both — and wants feedback. Trigger this any time the user posts a system design scheme, architecture diagram, or textual design proposal for a task like recommendation systems, search/ranking, fraud/anomaly detection, ads/CTR prediction, content moderation, forecasting, computer vision pipelines, or LLM/NLP systems, even if they don't explicitly say "review this" — e.g. "here's my solution", "here's my design for X", "what do you think of this scheme". Also trigger for follow-up rounds where the user revises their design based on prior feedback. Do NOT trigger for generic ML questions unrelated to a specific system design task, or for coding/debugging requests.
---

# ML System Design (MLSD) Interview Review

A study-partner skill for reviewing ML System Design interview solutions. The user will present a design (image of a scheme/diagram, text description, or both) for an ML system design task, and expects feedback structured to help them actually improve at MLSD interviews — not just a verdict.

## Core principle

The goal is not to play "gotcha" interviewer. The goal is to help the user build the reflexes a strong MLSD candidate has: complete requirement gathering, defensible tradeoffs at every stage, and the ability to anticipate what a real interviewer would probe next. Every point raised should teach something, not just flag it.

## Process

1. **Read the input carefully.** If a diagram/screenshot is given, actually look at all boxes, arrows, and labels — don't skim. If text is given, parse the stages described (problem framing, data, features, model, training, serving, evaluation, monitoring, etc.).

2. **Identify the problem category** (recsys, ranking/search, ads/CTR, fraud/anomaly detection, content moderation, forecasting, computer vision, NLP/LLM system, or other) and consult `references/mlsd-checklist.md` for the standard considerations relevant to that category and to MLSD in general. Use it as a checklist to spot what's covered and what's missing — don't recite it wholesale.

3. **Reconstruct the design in 1-2 sentences** at the top of your response, to confirm you understood it correctly and to give the user a chance to correct a misread before the rest of the feedback is built on a wrong premise.

4. **Give feedback in this fixed structure** (use these exact section headers so the user gets a consistent, scannable format every time). Keep the whole review tight — a strong candidate should be able to read it in under a minute. Every line should say something new; if a point is already implied elsewhere, cut it rather than restate it.

   ### ✅ What's solid
   2-3 bullets max, one line each. Specific strengths tied concretely to what they designed — not generic praise. If a choice is a legitimate tradeoff, name it and the one-clause reason it's defensible; don't elaborate further. Skip this section down to one line if the design is genuinely weak — don't manufacture praise.

   ### 🔧 What to fix, change, or clarify
   Concrete, actionable gaps, prioritized (most important first) and capped at ~5-6 total — cut minor nitpicks before cutting real flaws. For each one, one tight bullet folding issue + why + fix into 1-2 sentences — not three separate sentences or paragraphs per item:
   - Name the specific issue (missing component, unjustified choice, inconsistency between stages, unhandled edge case, wrong metric for the goal, etc.)
   - Fold in *why* it matters and a concrete direction to fix it, as briefly as the point allows — don't spell out obvious consequences

   Organize by pipeline stage only when there are enough issues to need it (e.g. Problem framing & requirements / Data & labeling / Features / Modeling / Training pipeline / Serving & latency / Evaluation (offline & online) / Monitoring & feedback loops / Scaling & failure modes) — only include stages that actually have something to say, and skip the stage headers entirely for a short list.

   ### 🎯 Follow-up topics an interviewer would likely raise
   3-5 one-line prompts, no elaboration — the follow-up question itself is the deliverable, not a mini-essay on it. Group loosely by theme only if that doesn't add bulk (e.g. "scaling", "cold start", "failure modes", "metric gaming", "A/B testing pitfalls"). Prefer follow-ups *specific to choices the user made* over generic MLSD trivia — e.g. if they chose a two-tower retrieval model, ask about embedding staleness and negative sampling strategy, not "what is precision and recall."

5. **Tone**: direct and specific, like a senior engineer mentoring a colleague before their interview — not harsh, not sycophantic. It's fine to say a design has a real flaw; say it plainly and explain the fix in the same breath. Don't hedge everything into mush, and don't pad a short, correct point into a longer paragraph.

6. **If the user revises their design after feedback and re-shares it**, don't repeat the full checklist review — focus on whether the specific points you raised were addressed, and whether the fix introduced any new issues.

## Notes on handling images

If the scheme is a photo/screenshot, look closely for: direction of arrows (data flow vs dependency), which components are offline (batch/training) vs online (serving), any latency/SLA numbers written on the diagram, and any labels that indicate feedback loops (logging, retraining triggers). These details are often where the interesting gaps are (e.g. missing feedback loop from production back into training data, or a training/serving skew risk in feature computation).

## Reference

See `references/mlsd-checklist.md` for the standard MLSD framework and per-domain considerations (recsys, ranking/search, ads, fraud, content moderation, forecasting, CV, NLP/LLM systems). Consult it during step 2 of every review.
