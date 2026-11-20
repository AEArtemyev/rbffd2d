# Engineering Principles

## Communication

- Write concise, direct, human prose in plain English.
- Challenge technical mistakes and explain trade-offs.
- Communicate uncertainty, assumptions, and limitations when they matter.
- Use ASCII for all newly written text unless explicitly instructed otherwise.
- Preserve existing non-ASCII content unless the task requires changing it.

## Engineering

- Prefer simple, cohesive designs over unnecessary abstractions.
- Favor composition over implementation inheritance when appropriate.
- Make failures explicit rather than silently hiding them.
- Keep changes focused on the requested task. Avoid unrelated refactoring.
- Report design problems and technical debt without fixing them unprompted.
- Follow existing project conventions unless explicitly instructed otherwise.

## Programming Style

- Prefer simple, imperative, and procedural code.
- Favor plain data structures, explicit control flow, and focused functions.
- Make data flow, state changes, and error handling easy to follow.
- Avoid unnecessary abstraction, indirection, and hidden behavior.
- Use OOP, functional programming, and other paradigms only when they
  clearly improve readability, correctness, or maintainability.
- Favor the pragmatism of experienced C programmers while respecting
  the idioms and conventions of the language and codebase.

## Workflow

- Understand the relevant code before modifying it.
- Reproduce bugs before fixing them when practical.
- Prefer regression tests for bug fixes to prevent the original bug from recurring.
- Do not add tests merely to reflect code changes; tests should protect meaningful behavior against plausible regressions.
- Avoid testing behavior already guaranteed by frameworks, libraries, or language constructs.
- Use red/green TDD when it meaningfully improves the implementation.
- Run relevant tests and checks after making changes.
- Investigate failures rather than masking or ignoring them.
- Make routine, reversible decisions independently.
- Ask for clarification when consequential design decisions remain unresolved.

## Git

- Preserve unrelated user changes.
- Do not pull, switch branches, commit, push, or open pull requests
  without explicit instruction.
- Follow the Conventional Commits specification for commit messages.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. Read `docs/agents/issue-tracker.md` before tracker operations.

### Triage labels

Use the default triage labels. Read `docs/agents/triage-labels.md` before applying triage labels.

### Domain docs

Use a single-context layout. Read `docs/agents/domain.md` before exploring domain concepts or decisions.
