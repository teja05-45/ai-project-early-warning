def get_top_risk_drivers(shap_values, feature_names, top_k=5):
    # Handle binary classifier SHAP output
    if isinstance(shap_values, list):
        shap_array = shap_values[1]  # positive class
    else:
        shap_array = shap_values

    shap_row = shap_array[0]

    contributions = dict(zip(feature_names, shap_row))

    return sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

def compare_driver_change(shap_current, shap_previous, feature_names, top_k=5):
    """
    Compare SHAP contributions between two sprints
    """

    # Handle binary classifier output
    if isinstance(shap_current, list):
        shap_current = shap_current[1]
        shap_previous = shap_previous[1]

    current = shap_current[0]
    previous = shap_previous[0]

    deltas = current - previous

    changes = dict(zip(feature_names, deltas))

    return sorted(
        changes.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]
