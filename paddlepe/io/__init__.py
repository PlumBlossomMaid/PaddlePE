"""Format I/O for .f0, .csv, .pv, .tsv files.

Auto-detection by file suffix.
"""

from paddlepe.io.reader import read, read_csv, read_f0, read_pv, read_tsv
from paddlepe.io.writer import write, write_csv, write_f0

__all__ = [
    "read",
    "read_f0",
    "read_csv",
    "read_pv",
    "read_tsv",
    "write",
    "write_f0",
    "write_csv",
]
