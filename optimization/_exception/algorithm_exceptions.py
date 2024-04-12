class ObjectError(Exception):
    """Exception raised when 'obj' is not Iterable."""

class DrawObjectError(Exception):
    """Exception raised when len('obj') is more than 3."""