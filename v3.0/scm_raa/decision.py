def assign_layer(raa_diag: dict) -> str:
    """
    Convert RAA status into an information layer.
    """
    status = raa_diag.get("status", "").lower()

    if status == "strong":
        return "foreground"
    if status == "weak":
        return "midground"
    return "background"


def decide_action(layer: str) -> str:
    """
    Decide what to do with each information layer.
    """
    if layer == "foreground":
        return "main_analysis"
    if layer == "midground":
        return "directed_exploration"
    return "control_reference"


def scm_decision_record(domain: str, x_col: str, y_col: str, raa_diag: dict) -> dict:
    """
    Build a final decision record for one CRTT + RAA result.
    """
    layer = assign_layer(raa_diag)
    action = decide_action(layer)

    return {
        "domain": domain,
        "x_col": x_col,
        "y_col": y_col,
        "status": raa_diag.get("status"),
        "confidence": raa_diag.get("confidence"),
        "layer": layer,
        "action": action,
        "delta_aic": raa_diag.get("delta_aic"),
        "cv_gain": raa_diag.get("cv_gain"),
        "iqr_frac": raa_diag.get("iqr_frac"),
        "n": raa_diag.get("n"),
        "veredicto": raa_diag.get("veredicto"),
    }
