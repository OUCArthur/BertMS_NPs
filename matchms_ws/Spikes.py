from deprecated.sphinx import deprecated
from matchms_ws.Fragments import Fragments


@deprecated(version='0.14.0', reason="Class was renamed to ``Fragments``.")
class Spikes(Fragments):
    """Deprecated! Please use :class:`~matchms.Fragments` instead."""
