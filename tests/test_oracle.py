"""The reference oracle, and the cost of being correct.

The uploaded circuit hit its depth target of 726 while marking one of
1097 specified pixels. These tests pin the other half of that sentence:
what the same specification costs when it is actually implemented.
"""
import unittest

try:
    from benchmarks.oracle import (GRID, build_oracle, cube_cover,
                                   disjoint_rectangles, encode, logo_predicate,
                                   marked_pixels)

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_QISKIT = False

from qem_auditor.reversible import audit_oracle


@unittest.skipUnless(HAVE_QISKIT, "needs qiskit to build the circuit")
class SpecificationTest(unittest.TestCase):

    def test_the_four_shapes_make_the_pixel_count_the_qmod_claimed(self):
        """The one thing in the uploaded submission that was right."""
        self.assertEqual(len(marked_pixels()), 1097)

    def test_rectangles_are_disjoint_and_cover_exactly(self):
        covered = set()
        for x0, x1, y0, y1 in disjoint_rectangles():
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.assertNotIn((x, y), covered, "rectangles overlap")
                    covered.add((x, y))
        self.assertEqual(covered, set(marked_pixels()))

    def test_cubes_are_disjoint_and_cover_exactly(self):
        """Disjointness is load-bearing, not tidiness: the oracle applies
        one phase per matching cube, so two cubes matching the same input
        would cancel to no phase at all."""
        covered = set()
        for cube in cube_cover():
            for x in range(cube.x_value, cube.x_value + cube.x_size):
                for y in range(cube.y_value, cube.y_value + cube.y_size):
                    self.assertNotIn((x, y), covered, "cubes overlap")
                    covered.add((x, y))
        self.assertEqual(covered, set(marked_pixels()))

    def test_a_cubes_controls_select_exactly_its_members(self):
        for cube in cube_cover()[:20]:
            for value in range(GRID * GRID):
                bits = encode(value)
                selected = all(bits[q] == b for q, b in cube.controls())
                self.assertEqual(selected, cube.contains(*divmod(value, GRID)))


@unittest.skipUnless(HAVE_QISKIT, "needs qiskit to build the circuit")
class ReferenceOracleTest(unittest.TestCase):

    def test_it_implements_its_specification_on_every_input(self):
        report = audit_oracle(
            build_oracle(),
            predicate=lambda v: logo_predicate(*divmod(v, GRID)),
            n_inputs=GRID * GRID, encode=encode, ancillas=())
        self.assertTrue(report.matches_specification, report.format_report())
        self.assertEqual(len(report.marked), 1097)

    def test_it_uses_no_ancillas(self):
        """Which is why the uploaded circuit's fatal defect -- ancillas
        left entangled on all 4096 inputs -- cannot occur here. There is
        nothing to uncompute, so there is no uncompute to get wrong."""
        self.assertEqual(build_oracle().num_qubits, 12)

    def test_a_wrong_specification_is_reported_rather_than_absorbed(self):
        """The oracle is built for one predicate and audited against
        another. A checker that passed this would be comparing the
        circuit with itself."""
        report = audit_oracle(
            build_oracle(),
            predicate=lambda v: logo_predicate(*divmod(v, GRID)) or v == 0,
            n_inputs=GRID * GRID, encode=encode, ancillas=())
        self.assertFalse(report.matches_specification)
        self.assertEqual(report.false_negatives, frozenset({0}))
