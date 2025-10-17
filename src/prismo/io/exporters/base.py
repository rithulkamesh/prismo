"""
Base exporter class for data export functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class Exporter(ABC):
    """
    Abstract base class for data exporters.

    All exporters must implement methods for exporting different types of data
    (fields, spectra, S-parameters, etc.) to their specific format.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize exporter.

        Parameters
        ----------
        output_dir : Path, optional
            Output directory for exported files. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export_fields(
        self,
        filename: str,
        fields: Dict[str, Any],
        coordinates: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export field data.

        Parameters
        ----------
        filename : str
            Output filename (without extension).
        fields : dict
            Dictionary of field arrays (Ex, Ey, Ez, Hx, Hy, Hz).
        coordinates : dict
            Dictionary of coordinate arrays (x, y, z).
        metadata : dict, optional
            Additional metadata to include.

        Returns
        -------
        Path
            Path to exported file.
        """
        pass

    @abstractmethod
    def export_spectrum(
        self,
        filename: str,
        frequencies: Any,
        spectrum: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export spectrum data.

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency array.
        spectrum : array
            Spectrum data.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported file.
        """
        pass

    @abstractmethod
    def export_sparameters(
        self,
        filename: str,
        frequencies: Any,
        sparameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export S-parameters.

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency array.
        sparameters : dict
            Dictionary of S-parameters (S11, S21, etc.).
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported file.
        """
        pass
