# Issue tracker: GitHub

Issues and specs live in AEArtemyev/rbffd2d on GitHub. Use the gh CLI
from this checkout; verify the target with `gh repo view --json nameWithOwner`
before creating issues.

## Operations

- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,body,labels`.
- Comment: `gh issue comment <number> --body-file <file>`.
- Label: `gh issue edit <number> --add-label "..." --remove-label "..."`.
- Close: post an explanation with `gh issue comment`, then run
  `gh issue close <number>`.

Use GitHub native sub-issues and issue dependencies where supported.
Check the installed gh command help before using version-specific flags.
If unavailable, put `Part of #<parent>` and `Blocked by: #<number>`
at the top of the child issue body and link children in the parent.
An issue is unblocked when every blocker is closed.

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read the GitHub issue.

## Pull requests as a triage surface

PRs as a request surface: no.

## Wayfinding

A map is an issue labelled `wayfinder:map`. Link child issues to the map
and label them `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling`, or `wayfinder:task`.

Choose the first open child in map order with no open blockers or assignee.
Claim it by assigning yourself. Resolve it by posting the result, closing
the child, and recording a result summary and link in the map's decisions.
