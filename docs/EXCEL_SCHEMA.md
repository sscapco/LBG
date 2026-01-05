# Excel Workbook Schema (`nodes_edges_governance.xlsx`)

The workbook is the workflow “source of truth”. It must contain:

## Sheet: `Nodes`

Required columns (after normalization in `core/governance_data.py`):

- `Stage`
- `Stage Name` (normalized to `Stage_Name`)
- `Step ID` (normalized to `Step_ID`)
- `Step name` (normalized to `Step_Name`)
- `Purpose (Brief one liner to outline purpose of the task)` (normalized to `Purpose`)
- `Description`

Optional columns:

- `Automatable` (truthy values: TRUE/YES/Y/1)
- `Automation_step` (string key referencing a tool in `core/automation_registry.py`, e.g. `validate_name`)

Notes:
- Step IDs are normalized to uppercase strings (e.g. `S21`).
- The **row order** of the Nodes sheet becomes the “canonical order” used for sorting next steps.

## Sheet: `Edges`

Required columns:

- `from`
- `to`
- `edge_type`
- `guard_condition`
- `Question/Probe` (normalized to `Question_Probe`)

Notes:
- Edge endpoints are normalized to uppercase strings.

