class HRRRError(Exception):
    """Base exception for HRRR extraction."""
    pass


class ConfigurationError(HRRRError):
    """Configuration related errors."""
    pass


class GRIBFileError(HRRRError):
    """GRIB file related errors."""
    pass


