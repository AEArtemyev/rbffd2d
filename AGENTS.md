# Engineering Principles

## Project goal

The project's goal is to support the numerical solution of partial differential
equations (PDEs) using RBF-FD in two-dimensional domains. It provides two core
capabilities for discretizing differential operators: finding neighboring nodes
and computing RBF-FD weights with polynomial augmentation.

These capabilities belong in one project because weight computation sometimes
requires expanding the stencil and searching for neighbors again. Keeping them
together allows reuse of the search code and the existing search tree, avoiding
the cost of rebuilding the tree during weight computation.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. Read `docs/agents/issue-tracker.md` before tracker operations.

### Triage labels

Use the default triage labels. Read `docs/agents/triage-labels.md` before applying triage labels.

### Domain docs

Use a single-context layout. Read `docs/agents/domain.md` before exploring domain concepts or decisions.
