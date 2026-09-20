"""End-to-end deployment of learned SEARCH-vs-REFINE policies.

Everything here is *deployment-side*: vectorized feature extraction that must match
`cold_start.growing.features.extract_features` column-for-column, a CRN-paired
evaluation harness, deployable recommenders, and the model-backed search policy.
Nothing in this package may read `GrowingState.mu`, a `Reservoir`, or any
`oracle_*` / `label_*` / `meta_*` column -- the harness computes hidden-truth
diagnostics itself, after the policy has acted.
"""
