import numpy as np
from matchms_ws.Fragments import Fragments
from matchms_ws.typing import SpectrumType


def select_by_mz(spectrum_in: SpectrumType, mz_from: float = 0.0,
                 mz_to: float = 1000.0) -> SpectrumType:
    """Keep only peaks between mz_from and mz_to (keep if mz_from >= m/z >= mz_to).

    Parameters
    ----------
    mz_from:
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to:
        Set upper threshold for m/z peak positions. Default is 1000.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."
    if len(spectrum.peaks) > 6:
        condition = np.logical_and(mz_from <= spectrum.peaks.mz, spectrum.peaks.mz <= mz_to)

        spectrum.peaks = Fragments(mz=spectrum.peaks.mz[condition],
                                intensities=spectrum.peaks.intensities[condition])
    # if len(spectrum.peaks) > 6:
    #     condition1 = np.logical_and(spectrum.get('precursor_mz')-17 <= spectrum.peaks.mz, spectrum.peaks.mz <= spectrum.get('precursor_mz')+17)

    #     spectrum.peaks = Fragments(mz=spectrum.peaks.mz[condition1],
    #                             intensities=spectrum.peaks.intensities[condition1])
    # if len(spectrum.peaks) > 6:
    #     liuxia = [0]
    #     for i in range(liuxia[-1]+1,len(spectrum.peaks)):
    #         if abs(spectrum.peaks.mz.round(2)[i] - spectrum.peaks.mz.round(2)[liuxia[-1]]) < 0.02:
    #             liuxia[-1] = liuxia[-1] if spectrum.peaks.intensities[liuxia[-1]] > spectrum.peaks.intensities[i] else i
    #         else:
    #             liuxia.append(i)
    #     spectrum.peaks = Fragments(mz=spectrum.peaks.mz[liuxia],
    #                             intensities=spectrum.peaks.intensities[liuxia])
    # if len(spectrum.peaks) > 6:
    #     spectrum.peaks = Fragments(mz=spectrum.peaks.mz[np.unique(spectrum.peaks.mz.round(2),return_index=True)[1]],
    #                             intensities=spectrum.peaks.intensities[np.unique(spectrum.peaks.mz.round(2),return_index=True)[1]])
                               
    return spectrum
