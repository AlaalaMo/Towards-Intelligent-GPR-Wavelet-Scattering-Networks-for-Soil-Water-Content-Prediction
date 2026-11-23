# AGENTS.md — AKELA Research Repository

## 1. Project Overview

This repository contains the research code supporting a scientific paper on Ground Penetrating Radar (GPR), signal processing, machine learning, deep learning, and soil water content (SWC) prediction.

The repository contains implementations of scientific methodologies, data-processing pipelines, model training procedures, and experimental evaluations used to produce published or submitted research results.

**The highest priority is preserving scientific correctness, reproducibility, and consistency with the research paper.**

Code is developed locally on Windows using PyCharm and Codex. Computationally expensive experiments are executed separately on a university HPC cluster using SLURM.

## 2. Primary Development Principles

Follow these principles in order of priority:

1. **Scientific reproducibility:** Preserve the exact computational methodology and experimental behavior.
2. **Correctness:** Ensure modifications do not introduce errors or unintended behavioral changes.
3. **Backward compatibility:** Maintain existing interfaces, inputs, outputs, and execution workflows.
4. **Maintainability:** Improve code organization, readability, and documentation without altering scientific behavior.
5. **Performance:** Consider optimizations only when they preserve numerical results and are explicitly authorized.

When these principles conflict, prioritize scientific reproducibility.

Do not assume that cleaner, shorter, or more efficient code is scientifically equivalent to the original implementation.

## 3. Python Environment

The local Python interpreter is:

`C:\Users\vince_lgsgqnj\.conda\envs\AKELA\python.exe`

- Use this interpreter for local Python execution and testing.
- Do not create or modify Python environments without approval.
- Do not install, remove, or upgrade dependencies without permission.
- Do not assume that Windows and HPC environments have identical dependencies.
- Preserve compatibility with the Python environment used for the original research experiments.

## 4. Python Coding Standards

Follow these conventions when creating or modifying Python code:

- Follow PEP 8.
- Use descriptive function, class, and variable names.
- Use type hints for new functions and methods.
- Use Google-style docstrings for functions, methods, and classes.
- Document arguments, return values, exceptions, assumptions, and side effects.
- Prefer functions with no more than four parameters.
- Prefer functions with no more than seven lines when practical.
- Split larger functions only when doing so preserves execution behavior and improves clarity.
- Apply object-oriented programming principles when appropriate.
- Avoid unnecessary abstractions and excessive class hierarchies.
- Use inline comments to explain scientific reasoning, mathematical operations, and non-obvious implementation decisions.
- Preserve existing public interfaces unless explicitly authorized to change them.

The function-length and parameter-count guidelines are preferences, not reasons to modify validated scientific implementations.

## 5. Scientific Reproducibility

**Do not modify scientific or computational behavior unless explicitly instructed.**

Protected components include:

- Mathematical equations and numerical algorithms.
- Signal-processing operations and transformations.
- Wavelet configurations and feature-extraction procedures.
- Data preprocessing, filtering, and normalization.
- Dataset selection, splitting, and sampling procedures.
- Machine learning and deep learning model architectures.
- Model initialization and random seeds.
- Loss functions and optimization algorithms.
- Hyperparameters, training schedules, and stopping criteria.
- Evaluation metrics and statistical calculations.
- Input/output formats and intermediate representations.
- Numerical precision, array shapes, and operation ordering.

Do not replace an implementation with an apparently equivalent alternative without approval.

Floating-point arithmetic, operation ordering, random-number generation, and library behavior can affect scientific results.

When a requested change could alter experimental results:

1. Identify the potential behavioral differences.
2. Explain the implications for reproducibility.
3. Request explicit approval before implementing the change.
4. Preserve the original implementation until approval is granted.

Never silently correct suspected scientific errors. Report them separately.

## 6. Code Modification Rules

- Modify only files directly relevant to the requested task.
- Preserve existing executable behavior unless explicitly authorized otherwise.
- Do not refactor unrelated functions.
- Do not rename functions, classes, parameters, or variables unnecessarily.
- Do not modify function signatures without approval.
- Do not reorganize execution order without approval.
- Prefer small, incremental patches over rewriting entire files.
- Avoid large PowerShell scripts or here-strings that may exceed Windows command-length limits.
- Do not delete existing code without explicit authorization.

### Documentation-Only Tasks

When asked to document code:

- Add or improve Google-style docstrings.
- Add inline comments only where they improve understanding.
- Explain the actual implementation rather than intended or assumed behavior.
- Document scientific assumptions only when supported by the code or provided research context.
- Do not modify executable statements.
- Do not change imports, signatures, algorithms, or control flow.
- Preserve existing useful comments.
- Verify that the Git diff contains only documentation changes.

### Refactoring Tasks

When asked to refactor code:

- Analyze the original implementation before making changes.
- Identify potential behavioral differences.
- Preserve numerical calculations and execution order.
- Preserve existing interfaces and data formats.
- Make incremental changes.
- Validate equivalence using available tests and representative inputs.
- Report any differences, including numerical discrepancies.

If behavioral equivalence cannot be established, explicitly state that the refactoring has not been fully validated.

## 7. Testing and Validation

Testing is required whenever practical.

- Use the configured AKELA Python interpreter.
- Run lightweight unit tests locally.
- Use `python -m pytest` when pytest is available.
- Perform syntax validation after modifying Python files.
- Avoid executing full training pipelines or computationally expensive simulations locally without approval.
- Do not modify datasets or experimental results during testing.
- Do not change reference outputs to make tests pass.
- Preserve existing test cases and expected outputs.

### Reproducibility Validation

For changes affecting executable code:

- Compare outputs before and after modifications using identical inputs.
- Preserve random seeds and experimental configurations.
- Check numerical equivalence using appropriate tolerances.
- Report any observed numerical differences.
- Distinguish between syntax validation, unit testing, and full experimental reproduction.

A successful syntax check or unit test does not establish full scientific reproducibility.

Do not claim that research results are preserved unless sufficient validation has been performed.

## 8. HPC and SLURM Compatibility

The repository is developed locally and executed on a Linux-based HPC cluster using SLURM.

- Preserve Linux compatibility.
- Preserve existing SLURM execution workflows.
- Do not modify SLURM resource allocations without approval.
- Do not submit, cancel, or modify HPC jobs automatically.
- Do not hardcode Windows-specific paths into HPC execution code.
- Preserve existing environment variables and execution assumptions.
- Do not modify or delete generated simulation outputs.
- Do not assume that local Windows testing reproduces HPC behavior.

HPC-specific functionality must be validated separately in the appropriate execution environment.

## 9. Git and Branch Management

The designated Codex development branch is:

`codex-development`

### Branch Rules

- Perform all code modifications on `codex-development`.
- Verify the active Git branch before modifying files.
- Do not modify code while on `main`.
- Do not switch branches automatically without permission.
- If the active branch is not `codex-development`, stop and report the issue.

### Git Restrictions

- Do not commit changes without explicit approval.
- Do not push changes to GitHub without explicit approval.
- Do not merge branches without explicit approval.
- Do not rewrite Git history.
- Do not use destructive Git commands without approval.
- Do not modify `.gitignore` unless explicitly requested.
- Do not commit datasets, credentials, generated results, or Python environments.

After modifications, inspect the Git diff and report all changed files.

The `main` branch represents the stable research implementation. Changes must be reviewed and validated before merging.

## 10. Task Execution Workflow

For each requested task:

### Before Modification

1. Identify the relevant files and functions.
2. Understand the existing implementation.
3. Determine whether the requested changes could affect scientific behavior.
4. Verify that the active branch is `codex-development`.

### During Modification

1. Modify only the necessary files.
2. Preserve existing computational logic.
3. Apply changes incrementally.
4. Avoid unrelated improvements.
5. Maintain existing interfaces and dependencies.

### After Modification

1. Inspect the Git diff.
2. Verify that no unintended changes were introduced.
3. Run appropriate lightweight validation.
4. Report any unverified behavior or potential reproducibility concerns.
5. Do not commit or push changes without approval.

## 11. Task Completion Report

After completing a task, provide:

1. **Files modified:** List all changed files.
2. **Functions modified:** Identify new and changed functions.
3. **Changes made:** Summarize the modifications.
4. **Scientific impact:** State whether computational behavior was changed or intended to remain unchanged.
5. **Validation:** List tests actually executed and their results.
6. **Reproducibility:** Identify any remaining uncertainty about equivalence with the original implementation.
7. **Git status:** Confirm that no unauthorized commits, pushes, or merges occurred.

Never claim that scientific behavior is unchanged solely because the modification appears minor.

## 12. General Restrictions

Unless explicitly requested:

- Do not change scientific methodology.
- Do not modify research datasets.
- Do not change model hyperparameters.
- Do not alter experimental configurations.
- Do not modify reference results.
- Do not perform unrelated refactoring.
- Do not install dependencies.
- Do not execute expensive experiments.
- Do not commit or push changes.

**The primary objective is to improve the quality, documentation, and maintainability of the research code while preserving the scientific implementation used to produce the paper's experimental results.**