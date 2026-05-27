from __future__ import annotations

import warnings

__author__ = 'Zhen Huang'
__email__ = 'zhen.victor.huang@gmail.com'
__version__ = '1.2.2'


warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
