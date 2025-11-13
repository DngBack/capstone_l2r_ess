# Table 3: Group Errors Comparison

| Method | Head Error (r=0.0) | Tail Error (r=0.0) | Gap (r=0.0) | Head Error (r=0.4) | Tail Error (r=0.4) | Gap (r=0.4) |
| --- | --- | --- | --- | --- | --- | --- |
| CE-only (Balanced) | 0.3564 | 0.6868 | 0.3305 | 0.2199 | 0.5575 | 0.3376 |
| CE-only (Worst-group) | 0.4776 | 0.5495 | 0.0719 | 0.3220 | 0.4143 | 0.0922 |
| LogitAdjust-only (Balanced) | 0.6114 | 0.4825 | -0.1289 | 0.4694 | 0.3146 | -0.1548 |
| BalSoftmax-only (Balanced) | 0.5915 | 0.5070 | -0.0844 | 0.4287 | 0.3340 | -0.0948 |
| BalSoftmax-only (Worst-group) | 0.5404 | 0.5304 | -0.0100 | 0.3686 | 0.3605 | -0.0081 |
| Uniform 2-Experts (Balanced) | 0.5848 | 0.4570 | -0.1278 | 0.3722 | 0.3708 | -0.0014 |
| Uniform 3-Experts (Balanced) | 0.6078 | 0.4482 | -0.1596 | 0.1653 | 0.5547 | 0.3894 |
| Uniform 2-Experts (Worst-group) | 0.4976 | 0.4818 | -0.0158 | 0.3122 | 0.3277 | 0.0155 |
| Uniform 3-Experts (Worst-group) | 0.4833 | 0.4751 | -0.0083 | 0.3005 | 0.3088 | 0.0083 |
| MoE (Gating) (Balanced) | 0.5614 | 0.4607 | -0.1008 | 0.4121 | 0.2835 | -0.1286 |
| MoE (Gating) (Worst-group) | 0.4828 | 0.4957 | 0.0129 | 0.2950 | 0.3299 | 0.0350 |

