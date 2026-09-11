# Project Rules & Guidelines

## 1. Version Control Automation (Mandatory)
- **Commit & Push After Every Change:** After completing any modification, refactor, bug fix, or user-requested change, automatically stage all modified files, commit with a concise and descriptive message, and push to GitHub (`git push origin main`).

## 2. Architecture & Code Quality
- **Separation of Concerns:** Keep indexing, data processing, and inference/API logic clean and modular.
- **Fast & Responsive:** Maintain on-demand streaming and responsive UI with sub-second performance.
- **Type Hinting & Error Handling:** Use Python type hints, Pydantic validation, and descriptive FastAPI `HTTPException` responses.
