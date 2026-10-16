"""Custom exceptions for NeuroScan AI system."""

class NeuroScanBaseException(Exception):
    """Base exception for all NeuroScan AI domain errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetNotFoundError(NeuroScanBaseException):
    """Raised when the expected brain MRI dataset cannot be located."""
    pass


class ModelNotReadyError(NeuroScanBaseException):
    """Raised when attempting inference, evaluation, or triage on uninitialized or missing models."""
    pass


class CorruptImageError(NeuroScanBaseException):
    """Raised when an MRI image file fails integrity or decode checks."""
    pass


class InvalidConfigurationError(NeuroScanBaseException):
    """Raised when invalid hyperparameter or file path configuration is supplied."""
    pass


class InferenceError(NeuroScanBaseException):
    """Raised when forward pass or explainability computation fails."""
    pass


class RLTriageError(NeuroScanBaseException):
    """Raised during reinforcement learning triage policy operations."""
    pass
