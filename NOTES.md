# NOTES — "What's true now" knowledge system

## Requirements

- Python 3.9 or newer (tested on 3.11), plus `pip install -r requirements.txt`. The only dependency is `anthropic==0.121.0`.
- An **`ANTHROPIC_API_KEY` environment variable is required.** This is a paid Claude API.
  - Claude writes every member-facing answer.
  - Model: `claude-sonnet-5-5` with adaptive thinking at `low` effort.
  - Without a key, the system still runs: ingest and `conflicts()` are fully local, and every `answer()` returns a polite "please contact us". Accuracy is then near zero.
- Approximate cost:
  - About 6–11k input tokens per question (the 2k-token system prompt is cached) and 100–450 output tokens.
  - That is roughly **$0.02 per question**, or about $1 per 40-question run.
- Optional settings:
  - `KB_MODEL` (default `claude-sonnet-5-5`)
  - `KB_EFFORT` (default `low`)
  - `KB_THINKING=on|off`
  - `KB_TIMEOUT` (seconds per API call, default 20)

**Data is not included.** As requested, the supplied website snapshots are not in this repository. To run the scorer, copy the kit's `snapshots_zip/` folder (or an already unpacked `snapshots/0*` folder) into the repository root. Then run:

```
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

## How it works

**Ingest** is local, deterministic, and takes about 6 s for all five snapshots.

1. **Clean** every page.
   - Remove crawl noise that changes on every recrawl: captcha images, chat widgets, phone-country pickers, "external link" modals, "Handpicked / Recent News" teaser sections, and http/https link flips.
   - Remove footnote digits glued to values ("4.20%1" becomes "4.20%").
   - This cut the number of pages that look "changed" between crawls from about 615 to about 220.
2. **Version** every URL. Each distinct cleaned text is stored once, tagged with the snapshots in which it was the live copy. "What did the site say on date X" then becomes a filter.
3. **Classify** pages by type and authority:
   - official rate sheet > product page > landing page > promo (`-lp`, offers) > news > blog > calculator / PDF > listing.
   - Also extract dates: news and blog datelines, and the "Rates are effective as of MM/DD/YYYY" line under each rate table.
4. **Chunk** pages:
   - Text is split by headings, and small sections are merged.
   - Tables stay whole, together with their caption and effective-date line.
   - Chunks go into a pure-Python BM25 index.
5. **Extract facts.** Every % cell of every table, and short "label: value" lines, become structured facts with:
   - product class (auto loan, certificate, mortgage, …)
   - qualifiers: term, balance or loan-amount tier, model-year band (ranked so "2023-26" and "2024-27" align), jumbo / special / Roth / business / 1st Rate …
   - metric (APY, APR, dividend rate, rate)
   - value, effective date, page and authority.

**Answer.** Each `answer` call takes about 2–4 s and makes one Claude call (rarely two).

1. **Pick the snapshot.**
   - With `as_of`: the newest snapshot taken on or before that date.
   - Otherwise, a past date written in the question ("as of May 2026") is used.
   - Otherwise, the latest snapshot.
   - Pages that disappeared from that snapshot are kept as a low-ranked fallback and labelled "no longer on the site".
2. **Retrieve** about 10 evidence passages, using BM25 plus boosts:
   - authority, rate tables for rate questions, recency for dated news, and clock times for hours questions;
   - business pages are demoted unless "business" is asked.
   - News and blog posts are given whole, and short chunks get their neighbouring chunks.
3. **Add "checked facts"** computed by code:
   - Any table value in the evidence that an official page contradicts, with the current value and where the outdated copy sits.
   - For "how many X" questions, the most recent *dated* statement. Frequency votes fail here, e.g. "23 branches" appears most often but "22" is newest.
   - For "what changed" questions, the value of each relevant row at every crawl date.
4. **Claude writes a JSON answer** following explicit rules: current value only, exact qualifiers, list everything for broad questions, never ask back, concrete URLs and phone numbers, and status `answered | partial | not_listed`.
5. **Guards** (deterministic) check every answer:
   - Every number and URL must appear in the evidence.
   - A "not listed" answer must contain no numbers at all, as the scorer requires.
   - No questions back to the member, and no mention of "the excerpts".
   - No superseded values.
   - A rate question answered without any rate is turned into an honest "not listed".
   - A violation triggers one corrected retry.
6. **Sources:** the pages Claude cited, plus every evidence page that contains the answer's values, most authoritative first.
7. **Cache:** answers are cached per (question, snapshot), so repeated questions are identical and instant.

## How we decide what is "current"

1. Answer from the snapshot that matches the question's date (default: the newest crawl).
2. If two pages state the same fact (same product, qualifiers and metric) with different values, official rate and product pages win over promo pages, calculators, blogs and PDFs. Between official pages, the latest "effective as of" date wins.
3. A promotion that has ended, or a closed branch, is not current.
4. Facts that appear only in dated news releases (branch counts, awards, closures, hours) come from the most recent release. Older releases are history.
5. Qualifiers are different facts, not conflicts: term, tier, model-year band, jumbo vs regular, APY vs dividend rate, APR vs rate, personal vs business, purchase vs refinance, and Share Savings vs HSA vs money market.
6. If the site does not publish it, say so ("isn't listed — please contact us"), with no numbers.

## conflicts()

1. **Rate rows:** the same product, qualifiers and metric showing different values across pages or crawls. The winner is the newest crawl, then an official page, then the latest effective date. A rate row and its APY/APR twin are merged into one conflict.
2. **Prose claims that drift:** branch count, counties served, employees, Forbes ordinal. The most recent dated statement wins.
3. **Excluded on purpose:**
   - rows whose product cannot be identified from the table itself, unless compared with the same table in another crawl;
   - rows a table lists twice with different values (an unparsed qualifier);
   - facts that are no longer on the site.

About 46 conflicts are reported. Each gives topic keywords (e.g. "60-month auto loan APR"), all competing values, the sources, the resolution and the reason.

## Results we measured (Claude Sonnet 5.5)

| Set | Accuracy | Stale-trap hits | Consistency |
|---|---|---|---|
| Public practice set (40) | 0.94 | 0 | 1.0 |
| Our own reworded / as-of / "what changed" set (`dev/paraphrase_eval.jsonl`, 47) | 1.00 | 0 | 1.0 |
| Harness selftest site (different, synthetic site; conflicts F1 1.0) | 1.00 | 0 | – |

p95 latency is about 4–5 s.

**Model A/B on the practice set:** Haiku 5.5 scored 0.83 accuracy (p95 3.4 s), Opus 5.5 low scored 0.87 (p95 6.0 s), and Sonnet 5.5 low scored 0.90–0.95 (p95 about 4.5 s). Sonnet had the best final score.

**Known misses:**
- `dup-005` has no `must_include`, so it scores 0 without the LLM judge.
- `dup-088` expects the 2023 Volunteer Time Off figure, but the site's newest figure is for 2024.

## Files

- `solution.py`: `build()` and the three contract methods.
- `dupaco_kb/clean.py`: noise removal and number normalisation.
- `dupaco_kb/pages.py`: page types, authority, dates.
- `dupaco_kb/store.py`: versions, chunking, fact extraction.
- `dupaco_kb/search.py`: BM25.
- `dupaco_kb/temporal.py`: which snapshot answers a question.
- `dupaco_kb/engine.py`: retrieval, hints, the answer pipeline.
- `dupaco_kb/llm.py`: prompt and Claude call.
- `dupaco_kb/guards.py`: answer checks.
- `dupaco_kb/conflicts.py`: contradiction detection.
- `../dev/`: our test tooling, not used by the grader.
