"""Comparator methods B0-B11 (plan section 12).

Every method receives the same legal input ceiling, the same disease split, the
same support and the same candidate list; only the learning algorithm differs.
A method that is allowed to use support genes gets them; a method that is not
applicable without support (PPI-RWR, quantum walk) reports "not applicable"
rather than a silent zero.
"""
from .registry import METHODS, get_method, list_methods  # noqa: F401
