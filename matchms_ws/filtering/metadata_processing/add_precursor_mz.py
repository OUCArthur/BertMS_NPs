import logging
from matchms_ws.utils import get_first_common_element


logger = logging.getLogger("matchms")


_default_key = "precursor_mz"
_accepted_keys = ["precursormz", "precursor_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def add_precursor_mz(spectrum_in):
    """Add precursor_mz to correct field and make it a float.

    For missing precursor_mz field: check if there is "pepmass"" entry instead.
    For string parsed as precursor_mz: convert to float.
    If precursor_mz is 0 or missing, use the m/z of the highest intensity peak.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    metadata_updated = _add_precursor_mz_metadata(spectrum.metadata, spectrum)
    spectrum.metadata = metadata_updated
    return spectrum


def _convert_precursor_mz(precursor_mz):
    """Convert precursor_mz to number if possible. Otherwise return None."""
    if precursor_mz is None:
        return None
    if isinstance(precursor_mz, str) and precursor_mz in _accepted_missing_entries:
        return None
    if not isinstance(precursor_mz, _accepted_types):
        logger.warning("Found precursor_mz of undefined type.")
        return None
    if isinstance(precursor_mz, str):
        try:
            return float(precursor_mz.strip())
        except ValueError:
            logger.warning("%s can't be converted to float.", precursor_mz)
            return None
    return precursor_mz


def _add_precursor_mz_metadata(metadata, spectrum=None):
    precursor_mz_key = get_first_common_element([_default_key] + _accepted_keys,
                                                metadata.keys())
    precursor_mz = metadata.get(precursor_mz_key)
    precursor_mz = _convert_precursor_mz(precursor_mz)

    # 如果 precursor_mz 存在且不为0，使用它
    if isinstance(precursor_mz, (float, int)) and precursor_mz > 0:
        metadata["precursor_mz"] = float(precursor_mz)
        for key in _accepted_keys:
            metadata.pop(key, None)
        return metadata

    pepmass = metadata.get("pepmass", None)
    if pepmass is not None:
        # Handle both single value and tuple/list formats
        pepmass_value = pepmass[0] if isinstance(pepmass, (tuple, list)) else pepmass
        if _convert_precursor_mz(pepmass_value) is not None:
            metadata["precursor_mz"] = float(pepmass_value)
            logger.debug("Added precursor_mz entry based on field 'pepmass'.")
            return metadata

    # 新增：如果 precursor_mz 为 0 或缺失，使用最高峰的 m/z
    if spectrum is not None and hasattr(spectrum, 'peaks') and spectrum.peaks is not None:
        if len(spectrum.peaks.mz) > 0 and len(spectrum.peaks.intensities) > 0:
            max_intensity_idx = spectrum.peaks.intensities.argmax()
            max_peak_mz = float(spectrum.peaks.mz[max_intensity_idx])
            metadata["precursor_mz"] = max_peak_mz
            logger.debug(f"Using highest intensity peak m/z ({max_peak_mz:.4f}) as precursor_mz")
            return metadata

    logger.debug("No valid precursor_mz found in metadata.")
    return metadata
