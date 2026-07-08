# Team GitHub Workflow

## Branching Strategy

- `main` holds releasable code only.
- Work starts on feature branches named `feature/[short-description]`.
- Branches are deleted after merge to keep the repo clean.

## Commit Message Convention

- Use these types: `feat`, `fix`, `docs`, `refactor`, `chore`.
- Format each commit as `[type]: [description]`.
- Keep the subject short and specific so the history is easy to scan.
- This supports clear collaboration and future changelog generation.

## PR Review Process

- Pull requests require at least one approval before merge.
- Reviewers focus on correctness, clarity, data integrity, and test coverage.
- Commit messages are reviewed as part of the PR checklist.

## Issue Tracking Approach

- Every feature or fix starts with a GitHub issue.
- Issues should include labels, assignees, and a clear description of done.
- Issues are closed when the related pull request is merged.

## Operating Note

- Keep collaboration work visible in issues and pull requests so context is not lost.