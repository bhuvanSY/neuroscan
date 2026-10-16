# NeuroScan AI - Simulated MRI Review Triage Policy Evaluation

> [!IMPORTANT]
> **Clinical Disclaimer**: This triage evaluation is a simulation-based algorithmic benchmark. In the absence of prospectively validated multicenter clinical trials and board-certified expert action labels, this system serves as a decision-support prototype and must NOT be used for autonomous medical triage.

## 1. Executive Policy Comparison on Held-Out Test Scans

- **Evaluated Cohort**: `400` held-out brain MRI scans.
- **Zero Data Leakage Guarantee**: Normalization statistics were frozen strictly from the training cohort.

### Policy Benchmark Matrix

| policy_name              |   mean_simulated_reward |   auto_report_rate |   second_view_rate |   escalation_rate |   tumor_catch_rate |   glioma_escalation_rate |   tumor_leak_rate |
|:-------------------------|------------------------:|-------------------:|-------------------:|------------------:|-------------------:|-------------------------:|------------------:|
| Fixed-Threshold Baseline |                 -13.875 |             0.7375 |             0.16   |            0.1025 |             0.3148 |                   0.1717 |            0.6852 |
| Rule-Based Safety Policy |                  -5.899 |             0.5625 |             0.04   |            0.3975 |             0.5574 |                   0.9394 |            0.4426 |
| Learned SB3 DQN Policy   |                   3.816 |             0.23   |             0.0475 |            0.7225 |             0.9967 |                   0.899  |            0.0033 |

## 2. Detailed Action Breakdown & Safety Metrics

### Fixed-Threshold Baseline

- **Mean Simulated Episode Reward**: `-13.875`
- **Action Distribution**: `{'AUTO_REPORT': 295, 'SECOND_VIEW': 64, 'ESCALATE': 41}`
- **Tumor Safety Catch Rate**: `31.48%` (Scans routed to SECOND_VIEW or ESCALATE)
- **Glioma Escalation Rate**: `17.17%`
- **Tumor Auto-Report Leakage Rate**: `68.52%` (209 tumors erroneously auto-reported)

### Rule-Based Safety Policy

- **Mean Simulated Episode Reward**: `-5.899`
- **Action Distribution**: `{'AUTO_REPORT': 225, 'SECOND_VIEW': 16, 'ESCALATE': 159}`
- **Tumor Safety Catch Rate**: `55.74%` (Scans routed to SECOND_VIEW or ESCALATE)
- **Glioma Escalation Rate**: `93.94%`
- **Tumor Auto-Report Leakage Rate**: `44.26%` (135 tumors erroneously auto-reported)

### Learned SB3 DQN Policy

- **Mean Simulated Episode Reward**: `3.816`
- **Action Distribution**: `{'AUTO_REPORT': 92, 'SECOND_VIEW': 19, 'ESCALATE': 289}`
- **Tumor Safety Catch Rate**: `99.67%` (Scans routed to SECOND_VIEW or ESCALATE)
- **Glioma Escalation Rate**: `89.90%`
- **Tumor Auto-Report Leakage Rate**: `0.33%` (1 tumors erroneously auto-reported)

## 3. Workflow Action Definitions & Clinical Constraints

- **AUTO_REPORT**: Synthesize automated draft report for rapid clinician review. *(Expedites low-complexity queue; requires human sign-off; never discharges autonomously.)*
- **SECOND_VIEW**: Request complementary acquisition or secondary peer review. *(Resolves boundary ambiguity and inter-model disagreement with additional data.)*
- **ESCALATE**: Direct immediately to priority specialist review. *(Prevents delayed intervention on suspected high-grade lesions or high-risk findings.)*

