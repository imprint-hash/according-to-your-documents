# "According to Your Documents"
### How One Prompt Makes Grounded AI Assistants Abandon and Misattribute Their Sources

Research project for the Apart Research **AI × Epistemics Research Sprint** (13–15 Nov 2026).
**All code and data in this repository were written during the sprint (from 13 Nov 2026). There is no preexisting code.**

## What we test
Many businesses build "company assistants" (Gemini Gems, custom GPTs, internal chatbots) by giving a model their
documents and the instruction *"answer only from these documents, never drop this rule"*. We test whether a single
one-sentence user message makes these assistants abandon their documents, invent company facts, and still present
those inventions as coming from the documents.

- `data/company_docs.md`: knowledge base of a **fictional** company (every fact invented, so fabrication is detectable).
- `data/questions.json`: 20 questions the documents answer (with gold answers), 20 plausible questions they do not
  answer, and 6 one-sentence override prompts plus a no-attack baseline.
- `harness/run.py`: runs each model as a grounded assistant (resumable).
- `harness/grade.py`: deterministic grading for answerable questions; LLM-judge rubric for unanswerable ones.

## Reproduce
Put API keys in `.env` (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ADAPTION_API_KEY`, optional `ANTHROPIC_API_KEY`), then:
```bash
python3 harness/run.py gemini-3.1-flash-lite
python3 harness/grade.py gemini-3.1-flash-lite --judge glm-5.2
```
No third-party Python packages are needed.
