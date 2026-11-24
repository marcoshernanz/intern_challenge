# AGENTS.md

This file provides instructions for AI agents (and human developers) working on the VLSI Placement Challenge repository.

## 1. Project Overview

The goal is to implement a **differentiable overlap loss function** (`overlap_repulsion_loss`) in `placement.py`. This function guides a gradient-based optimizer to place cells without overlapping.

- **Key Constraint:** The loss function MUST be fully differentiable.
- **Primary Metric:** Minimizing overlap (target: 0.0).
- **Secondary Metric:** Minimizing wirelength.

## 2. Build, Run, and Test

### Dependencies
This project uses standard Python tools.
- **Core:** `torch`
- **Visualization:** `matplotlib`
- **Linting:** `ruff`

### Commands

| Task | Command | Description |
|------|---------|-------------|
| **Run Demo** | `python placement.py` | Runs optimization on a default problem, logs loss, creates `placement_result.png`. |
| **Run All Tests** | `python test.py` | Runs the full suite of 12 test cases. |
| **Lint** | `ruff check .` | Checks for code quality issues. |
| **Format** | `ruff format .` | Formats code to standard style. |

### Running a Single Test
The `test.py` script runs all tests by default. To run a **single test case** (e.g., Test ID 1) for rapid iteration, use this Python one-liner in your terminal:

```bash
python -c "from test import run_placement_test; print(run_placement_test(1, 2, 20, 1001))"
```
*Args: `test_id`, `num_macros`, `num_std_cells`, `seed`*

## 3. Code Style & Conventions

### Python Guidelines
- **Style:** PEP 8 (enforced by `ruff`).
- **Indentation:** 4 spaces.
- **Imports:** Grouped: Standard Library -> Third Party (`torch`) -> Local.
- **Type Hints:** Required for function signatures.
  ```python
  def calculate_loss(features: torch.Tensor, scale: float) -> torch.Tensor:
  ```

### Naming Conventions
- **Functions/Variables:** `snake_case` (e.g., `compute_overlap`, `cell_area`).
- **Classes:** `PascalCase` (e.g., `CellFeatureIdx`).
- **Constants:** `UPPER_CASE` (e.g., `MAX_ITERATIONS`).

### Documentation
- Use **Google-style** docstrings.
  ```python
  def overlap_loss(features):
      """Calculates the overlap penalty.

      Args:
          features: Tensor of cell features [N, 6].

      Returns:
          Scalar tensor representing total overlap loss.
      """
  ```
- Comment complex tensor shape transformations (e.g., `# [N, 1] -> [N, N]`).

## 4. Critical Implementation Rules

### ⚠️ Differentiability (STRICT)
The optimizer uses PyTorch's autograd. Your loss function **cannot** break the computation graph.

1.  **NO** Python control flow (`if`, `for`, `while`) based on tensor values.
    *   ❌ Bad: `if overlap > 0: penalty += overlap`
    *   ✅ Good: `penalty = torch.relu(overlap)`
2.  **NO** Non-differentiable operations.
    *   ❌ Bad: `tensor.numpy()`, `tensor.item()`, `int(tensor)`
    *   ✅ Good: `torch.sum()`, `torch.mean()`, `torch.clamp()`
3.  **Vectorization:** Avoid loops. Use broadcasting.
    *   Example: To compare all N cells against each other, expand dimensions to `[N, 1, ...]` and `[1, N, ...]` to create `[N, N, ...]` comparison matrices.

### Data Structures
Reference `CellFeatureIdx` enum in `placement.py` for column indices:
- `features[:, 0]`: Area
- `features[:, 1]`: Num Pins
- `features[:, 2]`: **Center X** (Learnable)
- `features[:, 3]`: **Center Y** (Learnable)
- `features[:, 4]`: Width
- `features[:, 5]`: Height

## 5. Agent Workflow for Optimization
1.  **Analyze:** Read `placement.py` to understand the `features` tensor structure.
2.  **Implement:** Modify `overlap_repulsion_loss` in `placement.py`.
    *   Focus on `torch.relu` for detecting overlaps.
    *   Consider smooth approximations (like `logsumexp`) if gradients form plateaus, but `relu` is often sufficient for this task.
3.  **Iterate:** Run the **single test command** (see above) to check progress quickly.
4.  **Verify:** Run `python test.py` to ensure all scenarios pass (especially larger ones).
5.  **Refine:** If `overlap > 0` persists, increase the overlap penalty weight or adjust the loss formulation.
