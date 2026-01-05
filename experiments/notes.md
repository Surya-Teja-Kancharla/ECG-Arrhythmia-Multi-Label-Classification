## Key EDA Observations

- ECG signals exhibit high temporal variability and lead-specific morphology.
- Class distribution is severely imbalanced, with Normal and AF dominating.
- ~7% of samples contain multiple simultaneous arrhythmias.
- Certain arrhythmias frequently co-occur, justifying multi-label modeling.
- Lead correlations indicate shared cardiac activity but preserve unique patterns.
- Padding and temporal modeling are mandatory for downstream learning.

## PHASE 1 – ECG EDA SUMMARY
• Verified signal integrity and duration normalization
• Visualized 12-lead ECG morphology
• Identified severe class imbalance
• Quantified multi-label co-occurrence patterns
• Established need for weighted multi-label learning

# NOTE:
# Baseline correction is intentionally disabled by default
# to avoid altering clinically relevant ST-segment morphology.

print("""
Why stratified splitting matters (multi-label ECG):

- Random splits distort rare arrhythmia prevalence
- Labels co-occur (e.g., AF + PAC)
- Stratification preserves:
  • Per-class prevalence
  • Co-occurrence structure
  • Clinical realism

This project uses label-aware splitting prior to Phase 3 training.
""")

print("""
Cross-validation strategy (multi-label aware):

- Full K-fold is computationally expensive for ECG
- We rely on:
  • Dedicated validation split
  • Ensemble averaging
  • Threshold calibration

This achieves variance reduction without violating label dependencies.
""")


print("""
Phase 4 Summary:

✓ Loss functions compared conceptually (BCE vs Focal)
✓ Class imbalance quantified and addressed
✓ Multi-label stratification justified
✓ Cross-validation strategy explained
✓ Per-class threshold optimization implemented
✓ Significant post-training metric improvement achieved

Phase 4 focuses on DECISION QUALITY, not representation learning.
""")


Subset accuracy is intentionally low in multi-label ECG tasks due to its strict definition.
However, after per-class threshold optimization, we observe a measurable improvement, indicating better joint label consistency without retraining the model
