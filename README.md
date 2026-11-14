# According to Your Documents

Code and data for **"According to Your Documents": How One Prompt Makes Grounded AI Assistants Abandon and Misattribute Their Sources**
(Apart Research · AI x Epistemics sprint, Nov 2026).

Grounded assistants (Gemini Gems, custom GPTs, support bots) get a company's documents in their system prompt. We test what they do when a
customer asks something those documents don't answer: do they say so, or invent a company fact and present it as coming from the documents?

## What's here
| Path | What |
|---|---|
| `data/company_docs.md` | The fictional company knowledge base (Kestrel Haulage & Fulfilment) |
| `data/questions.json` | 20 answerable questions (gold answers), 20 unanswerable questions, 6 one-sentence attacks |
| `harness/run.py` | Runs a model under a condition: `--prompt hardened|typical|typical+{only,say,noinvent,hold}`, `--turns 1|2` |
| `harness/grade.py` | Deterministic grading for answerable questions; LLM judge for unanswerable ones |
| `harness/judge2.py`, `harness/classify_items.py` | Second-judge check; claim-type classification (invented specific / absence claim / supported) |
| `harness/analyze2.py` | Final analysis: two-judge strict/broad rates, Wilson CIs, paired bootstrap, ablation, figures |
| `results/raw/` | Every model response (`<model>@<condition>.jsonl`) |
| `results/graded/` | Judge labels (`<subject>__<judge>.jsonl`) |
| `results/final.json`, `results/figures/` | Numbers and figures used in the report |
| `report/` | Report source and PDF |

## Reproduce
```bash
python3 harness/run.py deepseek-v4-flash --prompt typical --turns 1
python3 harness/grade.py deepseek-v4-flash@typical-t1 --judge deepseek-v4-flash
python3 harness/analyze2.py
```
API keys go in a local `.env` (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ADAPTION_API_KEY`); it is git-ignored.

## Notes
- All company data is fictional. No personal data was collected.
- Question U06 (Sunday delivery) is excluded from the main analysis because the documents state "We are closed on Sundays"; results including it are reported as a sensitivity check.
- No optimised attack strings are released; the six attacks are short, well-known patterns.

MIT licence.
