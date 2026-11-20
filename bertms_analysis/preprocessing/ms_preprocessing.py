"""
Mass Spectrometry Preprocessing Functions

This module contains preprocessing functions for mass spectrometry data.
Functions extracted from run_层次聚类_modified.ipynb Cell 1.
"""

import matchms_ws.filtering as ms_filters


def metadata_processing(spectrum):
    """
    元数据处理

    处理质谱元数据，包括修复和统一化学结构信息。

    Parameters
    ----------
    spectrum : Spectrum
        输入的质谱对象

    Returns
    -------
    Spectrum
        处理后的质谱对象
    """
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.repair_inchi_inchikey_smiles(spectrum)
    spectrum = ms_filters.derive_inchi_from_smiles(spectrum)
    spectrum = ms_filters.derive_smiles_from_inchi(spectrum)
    spectrum = ms_filters.derive_inchikey_from_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_smiles(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchikey(spectrum)
    # 先解析 pepmass 字段（如果存在），再添加 precursor_mz
    spectrum = ms_filters.interpret_pepmass(spectrum)
    spectrum = ms_filters.add_precursor_mz(spectrum)
    return spectrum


def peak_processing(spectrum):
    """
    峰数据处理

    处理质谱峰数据，包括归一化、过滤低强度峰和选择 m/z 范围。

    Parameters
    ----------
    spectrum : Spectrum
        输入的质谱对象

    Returns
    -------
    Spectrum or None
        处理后的质谱对象，如果处理失败返回 None
    """
    try:
        spectrum = ms_filters.default_filters(spectrum)
        spectrum = ms_filters.normalize_intensities(spectrum)
        spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.001)
        spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)

        # 验证处理结果
        if (not hasattr(spectrum, 'peaks') or
            len(spectrum.peaks.intensities) == 0 or
            spectrum.peaks.intensities.max() <= 0):
            return None
        return spectrum
    except Exception as e:
        print(f"处理谱图时出错: {e}")
        return None


def is_valid_spectrum(spectrum):
    """
    检查谱图是否有效

    Parameters
    ----------
    spectrum : Spectrum
        待检查的质谱对象

    Returns
    -------
    bool
        如果谱图有效返回 True，否则返回 False
    """
    try:
        return (hasattr(spectrum, 'peaks') and
                hasattr(spectrum.peaks, 'intensities') and
                hasattr(spectrum.peaks, 'mz') and
                len(spectrum.peaks.intensities) > 0 and
                len(spectrum.peaks.mz) > 0 and
                spectrum.peaks.intensities.max() > 0)
    except:
        return False
