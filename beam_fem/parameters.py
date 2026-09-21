"""Beam/frame geometry, material parameters, and default connectivity."""

from dataclasses import dataclass

import numpy as np


class BeamParameters:
    """Container for beam geometry and material properties."""

    def __init__(self, length=10.0, num_elements=25, youngs_modulus=200e9, moment_inertia=1.0, density=7800.0):
        # Geometry
        self.length = length                    # Total beam length (m)
        self.num_elements = num_elements                # Number of finite elements
        self.num_nodes = self.num_elements + 1
        self.element_length = self.length / self.num_elements

        # Material properties
        self.youngs_modulus = youngs_modulus           # Young's modulus (Pa)
        self.moment_inertia = moment_inertia             # Moment of inertia (m^4)
        self.density = density                 # Material density (kg/m^3)

        # Derived properties
        self.node_positions = np.linspace(0, self.length, self.num_nodes)
        self.elements = self._create_element_connectivity()

    def _create_element_connectivity(self):
        """Create element connectivity matrix."""
        elements = []
        for i in range(self.num_elements):
            elements.append([i, i + 1])  # [start_node, end_node]
        return elements


@dataclass
class FrameworkNode:
    name: str
    x: float
    y: float


@dataclass
class FrameworkElement:
    start: int
    end: int
    youngs_modulus: float
    cross_section_area: float
    moment_inertia: float
    density: float


def default_framework_nodes():
    """
    Coordinates for p1...p4.

    The lecture image is schematic, so these coordinates define a simple
    representative geometry with p1 fixed at the lower left and p4 on the
    right support line.
    """
    return [
        FrameworkNode("p1", 0.0, 0.0),
        FrameworkNode("p2", 3.0, 0.6),
        FrameworkNode("p3", 2.0, 2.4),
        FrameworkNode("p4", 5.0, 3.0),
    ]


def default_framework_connectivity():
    """Five one-element beams in the simple framework."""
    return [
        (0, 1),  # p1-p2
        (0, 2),  # p1-p3
        (1, 2),  # p2-p3
        (1, 3),  # p2-p4
        (2, 3),  # p3-p4
    ]
