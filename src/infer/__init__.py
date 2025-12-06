"""
Inference utilities for single image comparison.

This module provides functions for loading models, running inference pipelines,
and visualizing results.
"""

from .loaders import (
    load_class_to_group,
    load_test_sample_with_image,
    load_image_from_infer_samples,
    load_ce_expert,
    load_all_experts,
    load_gating_network,
    load_plugin_params,
    DATASET,
    NUM_CLASSES,
    NUM_GROUPS,
    TAIL_THRESHOLD,
    DEVICE,
    EXPERT_NAMES,
    EXPERT_DISPLAY_NAMES,
    SPLITS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
)

from .pipeline import (
    BalancedLtRPlugin,
    paper_method_pipeline,
    our_method_pipeline,
    compute_rejection_thresholds_from_test_set,
    plot_ce_only_full_class_distribution,
    plot_full_class_distribution,
    visualize_comparison,
    OUTPUT_DIR,
)

__all__ = [
    # From loaders
    'load_class_to_group',
    'load_test_sample_with_image',
    'load_image_from_infer_samples',
    'load_ce_expert',
    'load_all_experts',
    'load_gating_network',
    'load_plugin_params',
    'DATASET',
    'NUM_CLASSES',
    'NUM_GROUPS',
    'TAIL_THRESHOLD',
    'DEVICE',
    'EXPERT_NAMES',
    'EXPERT_DISPLAY_NAMES',
    'SPLITS_DIR',
    'CHECKPOINTS_DIR',
    'RESULTS_DIR',
    # From pipeline
    'BalancedLtRPlugin',
    'paper_method_pipeline',
    'our_method_pipeline',
    'compute_rejection_thresholds_from_test_set',
    'plot_ce_only_full_class_distribution',
    'plot_full_class_distribution',
    'visualize_comparison',
    'OUTPUT_DIR',
]
