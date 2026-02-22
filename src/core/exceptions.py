class IngestionError(Exception):
    pass


class UnsupportedFormatError(IngestionError):
    pass


class FileTooLargeError(IngestionError):
    pass


class FileReadError(IngestionError):
    pass
