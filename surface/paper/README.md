# `surface/paper`: figure and table generators

These produce the Sections 2 to 4 diagnostics, which are read from the retention-experiment
receipts rather than from training cells. The paper sources themselves are not part of this
release; each script writes LaTeX fragments and PDFs into a directory it creates.

| file | output |
|---|---|
| `make_figures.py` | the tie-rate identity panel and the calibration-and-power panel. |
| `make_fig1_pptx.py` | the overview figure. Needs `python-pptx`. |
| `make_headtohead.py` | the live head-to-head table on MCP servers. |
| `make_replicates.py`, `make_rltable.py` | the replication and early RL-arm tables. |

Every number is read from a banked receipt rather than hard-coded, so a figure cannot drift
from the result it reports. The receipts are generated artefacts and are not in this
repository; where one is missing, `make_figures.py` falls back to the literals recorded
beside the function that draws the panel.
