# Domain Brief

<!--
Fill this file in before running `autoresearch init`. The AI agent reads it at
the start of every session to understand the problem context and what to try.
The more detail you provide, the better the agent's feature engineering will be.

Delete this comment block and replace the sections below with your own content.
-->

## What we are predicting

[One sentence: what is the target column and what does it represent?]

Example: "We are predicting whether a loan applicant will default (binary: good / bad credit risk)."

## Dataset description

[Describe the rows, the columns, and any domain-specific structure you know about.]

Example:
- Each row is a loan application.
- Columns include credit history, loan amount, duration, employment status, housing type, and personal demographics.
- The dataset has 1000 rows and 20 feature columns.

## Known signal sources

[What columns do you expect to be predictive? Why?]

Example:
- `credit_history` — applicants with existing paid-off credit are lower risk.
- `duration` — longer loan durations increase risk of default.
- `amount` — high loan amounts relative to income are a classic risk signal.

## Feature ideas to try

[Any transforms, interactions, or aggregations you think are worth exploring.]

Example:
- Ratio of `amount` to `duration` (monthly payment burden).
- Binned `age` buckets (young / middle-aged / senior).
- Interaction between `employment` status and `loan_purpose`.
- Target-encoded `housing` type (OOF to prevent leakage).

## What to avoid

[Columns that are likely leaky, redundant, or uninformative.]

Example:
- `foreign_worker` — low cardinality, likely spurious correlation.
- Any columns that encode the target directly.

## Success criteria

[What score would you consider a win? What is the baseline?]

Example: Baseline accuracy is 0.750. A good result is above 0.775.
