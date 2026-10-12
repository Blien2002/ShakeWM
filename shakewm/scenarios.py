"""Canonical names for completed current-catalog ShakeBench episodes.

Compatibility names are accepted only at ingestion. A suffix alone never
turns one of the retired physical designs into a current scenario.
"""

CURRENT_SCENARIOS = ("S1", "S2", "S3", "S4")
_CURRENT_INPUT_ALIASES = {"S1r": "S1", "S2w": "S2", "S4r": "S4"}
_LATENT_KEYS = {
    "S1": {"roll_hz", "axis_psi_rad", "amp_deg"},
    "S2": {"fx_hz", "fy_hz", "fz_hz", "h_rms", "v_rms"},
    "S3": {"f0_hz", "direction", "radius_m"},
    "S4": {"stroke_hz", "fn_hz", "psi_rad", "gamma_peak", "rho_h"},
}


def canonical_scenario(excitation):
    """Return a current name without modifying the source excitation record.

    Completed unversioned records must contain the current builder's latent
    signature. Retired low-frequency rocking/vehicle/floor designs cannot be
    relabeled just because they previously used the same short scenario ID.
    The importer retains its existing checksum binding to the original source.
    """
    params = excitation.get("mode_params", {})
    name = params.get("scenario", "unspecified")
    if excitation.get("mode") != "scenario_lines_v1":
        return name
    canonical = _CURRENT_INPUT_ALIASES.get(name, name)
    if canonical not in CURRENT_SCENARIOS:
        raise ValueError(f"scenario {name!r} has no current S1–S4 mapping")
    version = params.get("scenario_catalog_version")
    if version not in (None, "canonical_v2"):
        raise ValueError(f"unsupported scenario catalog {version!r}")
    latents = params.get("latents", {})
    if not _LATENT_KEYS[canonical].issubset(latents):
        raise ValueError(f"ambiguous or retired physical definition for scenario {name!r}")
    return canonical
