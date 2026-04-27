# Supply-Chain Resilience Dynamic Programming Project

This folder is the submission package for the merged PS7 + PS8 project.

## Main report
- `SupplyChainResilience_Playbook.pdf` (compiled report)
- `SupplyChainResilience_Playbook.tex` (LaTeX source)

## Code
- `execute_vfi.py` and `functions_vfi.py` (value function iteration)
- `execute_pfi.py` and `functions_pfi.py` (policy function iteration)

## Figures
- `value_function.png`
- `value_function_pfi.png`
- `domestic_sourcing_policy.png`
- `domestic_sourcing_policy_pfi.png`
- `subsidy_policy.png`
- `subsidy_policy_pfi.png`
- `next_period_capacity_policy.png`
- `next_period_capacity_policy_pfi.png`

## Build the report
From this folder, run:

```bash
latexmk -pdf SupplyChainResilience_Playbook.tex
```

This generates `SupplyChainResilience_Playbook.pdf`.
