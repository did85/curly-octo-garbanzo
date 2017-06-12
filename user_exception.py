class Error(Exception):
    """Base exception class."""
    pass


class CommandError(Error):
    """Exception related to parsing the CLI."""
    pass
