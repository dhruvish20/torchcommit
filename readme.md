# Project Proposal: TorchCommit — Versioned Experiment Tracker for PyTorch

---

## Overview

TorchCommit is a lightweight experiment tracking tool designed to seamlessly integrate with PyTorch training workflows. It captures and versions all relevant parameters, hyperparameters, system resource usage, and training metrics for each experiment run. The goal is to enable researchers and developers to track, compare, and understand how changes in model architecture or training configuration affect performance — without the overhead of existing complex tools.

---

## Motivation

In machine learning research and development, iterative experimentation with various model architectures, optimizers, hyperparameters, and training strategies is common. However:

Researchers often lose track of the exact settings and results for each run.

Existing tools can be heavy, require cloud setup, or enforce specific workflows.

Explaining why a model’s performance changed after tweaking parameters is non-trivial.

There is a lack of simple, local-first tools that also support intelligent explanation of experiment diffs.

TorchCommit aims to fill this gap by providing:

Transparent and reproducible tracking of PyTorch experiments

A simple CLI to commit, compare, and explain experiment runs

Storage of detailed metadata in human-readable, diff-friendly JSON

Integration with AI agents (LLMs) to explain experiment result changes

A local-first design, allowing offline use and easy integration with git

## Core Features
1. Automatic Parameter Extraction

- Extract model architecture details, optimizer settings, loss function, epochs, batch size, etc. from PyTorch training script.

- Record hardware and resource usage (CPU/GPU utilization, memory consumption, training duration).

2. Versioned Commit System

- Save experiment configurations, metrics, and checkpoints in a versioned local directory (.torchcommit/).

- Allow users to commit new experiment runs with descriptive comments.

- Enable rollback or re-run of any saved experiment.

3. Experiment Comparison & Explanation

- Provide CLI commands to diff two experiment commits, highlighting configuration and metric differences.

- Use Large Language Models (e.g., GPT-4) to generate natural language explanations about why performance changed between commits.

4. Resource Efficiency & Privacy

- Store all data locally by default, with optional support for remote backends later.

- Provide commands for cleaning up old or unwanted experiment data.

5. Extensible CLI Interface

- Commands like torchcommit commit, torchcommit compare, torchcommit explain, torchcommit clean.

- Human-readable output and JSON export for automation.

## Technical Approach
- Language & Framework: Python 3.9+, PyTorch

- Data Storage: JSON files for configs and metadata; PyTorch .pth files for model checkpoints.

- System Monitoring: Use psutil to log CPU/GPU usage and memory during training.

- Configuration Management: Use OmegaConf or Hydra to manage experiment configs.

- Diff & Explain: Use difflib or deepdiff for config diffs; integrate GPT-4 API or local LLM to explain experiment differences.

- CLI Interface: Use Typer or Click for a clean command line UX.

- Version Control: Optionally integrate with GitPython to link experiment commits with Git commits.

## Expected Outcomes

- A fully functional, easy-to-use CLI tool for experiment tracking with PyTorch.

- A reproducible experiment history that improves research productivity.

- Meaningful, AI-generated explanations that reduce the cognitive load of experiment analysis.

- Local-first design ensuring privacy and offline usability.

## Resources Needed
- Access to GPUs for training and testing (optional but helpful).

- OpenAI GPT API key or local LLM resources (for explanation feature).

- Guidance on best practices for system design and experimentation workflows.

## Possible Extensions (Future Work)
- Web or GUI dashboard to visualize experiment history and metrics.

- Integration with popular tracking tools (Weights & Biases, CometML).

- Support for other ML frameworks (TensorFlow, JAX).

- Collaborative multi-user experiment sharing with remote backends.


