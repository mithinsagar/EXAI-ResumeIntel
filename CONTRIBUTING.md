# Contributing to EXAI-ResumeIntel

Thank you for considering contributing to EXAI-ResumeIntel. This document outlines the process for contributing code, documentation, and bug reports.

**Maintainer:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Ways to Contribute

- Report bugs via [GitHub Issues](https://github.com/mithinsagar/EXAI-ResumeIntel/issues)
- Suggest new features or improvements
- Add support for additional job roles or skill domains
- Improve documentation
- Add tests for uncovered code paths
- Submit pull requests for open issues

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EXAI-ResumeIntel.git
   cd EXAI-ResumeIntel
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- **Formatter:** [Black](https://github.com/psf/black) with line length 100
- **Import sorter:** [isort](https://github.com/PyCQA/isort) with Black profile
- **Linter:** [Ruff](https://github.com/astral-sh/ruff)
- **Type checker:** [mypy](https://mypy-lang.org/) in strict mode for new modules

Run the full quality check before pushing:

```bash
make lint
make test
```

Or individually:

```bash
black core/ xai/ api/ training/ tests/
isort core/ xai/ api/ training/ tests/
ruff check core/ xai/ api/ training/ tests/
mypy core/ xai/ api/
```

## Testing

- Add tests for every new function or class in the appropriate `tests/test_*.py` module.
- Maintain overall coverage above 80%.
- Run tests locally before submitting:
  ```bash
  pytest tests/ -v --cov=core --cov=xai --cov=api
  ```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): short description

Optional longer body.
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Examples:
- `feat(ontology): add cybersecurity skill node`
- `fix(scorer): handle empty skill vector`
- `docs(api): document /analyze endpoint response schema`

## Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Run linting and tests locally.
3. Update relevant documentation (`docs/`, README, docstrings).
4. Open a pull request against `main` with a clear description referencing any related issue.
5. Fill in the PR template completely.
6. Wait for review. Address feedback with additional commits (do not force-push during review).

## Adding a New Skill Domain

To add a new skill or job role:

1. Add the canonical skill node to `SKILL_ONTOLOGY` in `core/ontology.py` with its aliases, parent nodes, and category.
2. Add the role weight profile to `ROLE_SKILL_WEIGHTS` in the same file (or edit `config/role_weights.json`).
3. Add unit tests in `tests/test_ontology.py` that verify:
   - The new skill is extractable from example text.
   - Parent propagation works correctly.
   - Role weights sum to a reasonable range.
4. Update the frontend role selector in `ui/index.html`.

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:

- Python version and OS
- Minimal reproduction steps
- Expected vs actual behaviour
- Full error traceback if applicable

## Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md). Explain the use case, proposed solution, and any alternatives considered.

## Code of Conduct

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md). In short: be respectful, constructive, and inclusive.

## License

By contributing to EXAI-ResumeIntel, you agree that your contributions will be licensed under the MIT License.
