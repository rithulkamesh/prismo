"""
Parser for Lumerical .fsp (FDTD project) files.

This module provides functionality to parse Lumerical FDTD Solutions
project files and convert them to Prismo format.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import json


@dataclass
class FSPGeometry:
    """Geometry object from FSP file."""

    name: str
    type: str  # 'rectangle', 'circle', 'polygon', etc.
    center: tuple
    size: tuple
    material: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FSPSource:
    """Source definition from FSP file."""

    name: str
    type: str  # 'mode', 'gaussian', 'plane_wave', etc.
    center: tuple
    size: tuple
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FSPMonitor:
    """Monitor definition from FSP file."""

    name: str
    type: str  # 'time', 'frequency', 'mode_expansion', etc.
    center: tuple
    size: tuple
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FSPProject:
    """
    Complete FSP project data.

    Attributes
    ----------
    filename : Path
        Original .fsp filename.
    geometries : List[FSPGeometry]
        Geometry objects.
    sources : List[FSPSource]
        Source definitions.
    monitors : List[FSPMonitor]
        Monitor definitions.
    simulation_region : dict
        Simulation domain parameters.
    materials : Dict[str, Any]
        Custom material definitions.
    metadata : dict
        Additional metadata.
    """

    filename: Path
    geometries: List[FSPGeometry] = field(default_factory=list)
    sources: List[FSPSource] = field(default_factory=list)
    monitors: List[FSPMonitor] = field(default_factory=list)
    simulation_region: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FSPParser:
    """
    Parser for Lumerical .fsp files.

    Lumerical FDTD Solutions saves projects as .fsp files, which are
    compressed archives containing XML and binary data. This parser
    extracts the relevant information for conversion to Prismo.

    Parameters
    ----------
    fsp_file : Path
        Path to .fsp file.
    """

    def __init__(self, fsp_file: Path):
        self.fsp_file = Path(fsp_file)

        if not self.fsp_file.exists():
            raise FileNotFoundError(f"FSP file not found: {fsp_file}")

        self.project: Optional[FSPProject] = None

    def parse(self) -> FSPProject:
        """
        Parse the .fsp file.

        Returns
        -------
        FSPProject
            Parsed project data.
        """
        # FSP files are compressed archives (similar to ZIP)
        # They contain XML files and binary data

        # This is a simplified parser - full implementation would:
        # 1. Extract .fsp archive
        # 2. Parse XML structure files
        # 3. Extract geometry definitions
        # 4. Extract source/monitor definitions
        # 5. Extract material assignments

        project = FSPProject(filename=self.fsp_file)

        # Placeholder parsing logic
        # Real implementation would use zipfile to extract and parse

        try:
            # Attempt to parse as text/XML directly (some FSP files)
            self._parse_xml_structure(project)
        except Exception as e:
            # Try archive extraction
            self._parse_archive(project)

        self.project = project
        return project

    def _parse_xml_structure(self, project: FSPProject) -> None:
        """Parse XML structure from FSP file."""
        # Placeholder - real implementation would parse actual FSP XML

        # Example structure parsing
        project.metadata["parser_note"] = (
            "FSP parser is a simplified implementation. "
            "Full parsing requires reverse engineering of Lumerical's format."
        )

    def _parse_archive(self, project: FSPProject) -> None:
        """Extract and parse FSP archive."""
        import zipfile

        try:
            with zipfile.ZipFile(self.fsp_file, "r") as zf:
                # List files in archive
                files = zf.namelist()
                project.metadata["archive_files"] = files

                # Look for structure files
                for filename in files:
                    if filename.endswith(".xml"):
                        # Parse XML file
                        with zf.open(filename) as f:
                            xml_content = f.read()
                            self._parse_xml_content(xml_content, project)

        except zipfile.BadZipFile:
            # Not a zip archive - might be binary format
            project.metadata["format"] = "binary"

    def _parse_xml_content(self, xml_content: bytes, project: FSPProject) -> None:
        """Parse XML content."""
        try:
            root = ET.fromstring(xml_content)

            # Extract geometries
            for geom_elem in root.findall(".//geometry"):
                geometry = self._parse_geometry_element(geom_elem)
                if geometry:
                    project.geometries.append(geometry)

            # Extract sources
            for source_elem in root.findall(".//source"):
                source = self._parse_source_element(source_elem)
                if source:
                    project.sources.append(source)

            # Extract monitors
            for monitor_elem in root.findall(".//monitor"):
                monitor = self._parse_monitor_element(monitor_elem)
                if monitor:
                    project.monitors.append(monitor)

        except ET.ParseError:
            pass

    def _parse_geometry_element(self, elem: ET.Element) -> Optional[FSPGeometry]:
        """Parse a geometry element from XML."""
        try:
            return FSPGeometry(
                name=elem.get("name", "unnamed"),
                type=elem.get("type", "unknown"),
                center=(0, 0, 0),  # Extract from XML
                size=(0, 0, 0),  # Extract from XML
                material=elem.get("material", ""),
            )
        except:
            return None

    def _parse_source_element(self, elem: ET.Element) -> Optional[FSPSource]:
        """Parse a source element from XML."""
        try:
            return FSPSource(
                name=elem.get("name", "unnamed"),
                type=elem.get("type", "unknown"),
                center=(0, 0, 0),
                size=(0, 0, 0),
            )
        except:
            return None

    def _parse_monitor_element(self, elem: ET.Element) -> Optional[FSPMonitor]:
        """Parse a monitor element from XML."""
        try:
            return FSPMonitor(
                name=elem.get("name", "unnamed"),
                type=elem.get("type", "unknown"),
                center=(0, 0, 0),
                size=(0, 0, 0),
            )
        except:
            return None

    def to_prismo_simulation(self):
        """
        Convert FSP project to Prismo Simulation.

        Returns
        -------
        Simulation
            Prismo simulation object configured from FSP file.
        """
        from prismo import Simulation

        if self.project is None:
            raise RuntimeError("Must parse FSP file first")

        # Extract simulation region
        sim_region = self.project.simulation_region

        # Create Prismo simulation
        # (Simplified - would need proper parameter mapping)
        sim = Simulation(
            size=(10e-6, 5e-6, 0),  # Placeholder
            resolution=50e6,
        )

        # Add geometries, sources, monitors
        # (Would require full conversion logic)

        return sim

    def export_summary(self, output_file: Optional[Path] = None) -> Path:
        """
        Export summary of FSP contents to JSON.

        Parameters
        ----------
        output_file : Path, optional
            Output file path.

        Returns
        -------
        Path
            Path to summary file.
        """
        if self.project is None:
            raise RuntimeError("Must parse FSP file first")

        if output_file is None:
            output_file = self.output_dir / f"{self.fsp_file.stem}_summary.json"

        summary = {
            "filename": str(self.fsp_file),
            "num_geometries": len(self.project.geometries),
            "num_sources": len(self.project.sources),
            "num_monitors": len(self.project.monitors),
            "geometries": [
                {"name": g.name, "type": g.type, "material": g.material}
                for g in self.project.geometries
            ],
            "sources": [{"name": s.name, "type": s.type} for s in self.project.sources],
            "monitors": [
                {"name": m.name, "type": m.type} for m in self.project.monitors
            ],
            "metadata": self.project.metadata,
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        return output_file
