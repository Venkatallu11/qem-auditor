"""The service surface, tested without any protocol machinery.

The MCP binding is a thin shell over these functions, so testing them
directly is testing the service. What is checked here is mostly that the
refusals survive the trip through a JSON-shaped boundary -- a service
that answered "GO" where the library said "SKIP" would be worse than no
service.
"""
import unittest

from qem_auditor.service import (TOOLS, budget_from_device, check_feasibility,
                                 compare_two_methods, falsify_claim,
                                 recommend_mitigation)


class SurfaceTest(unittest.TestCase):

    def test_the_advertised_manifest_matches_the_real_tools(self):
        """A manifest that drifts from the implementation is how a service
        starts promising things it does not do."""
        import mcp_server
        advertised = {tool["name"] for tool in mcp_server.describe()["tools"]}
        self.assertEqual(advertised, set(TOOLS))
        for tool in mcp_server.describe()["tools"]:
            self.assertTrue(tool["description"].strip())

    def test_no_tool_here_talks_to_a_quantum_computer(self):
        """The division of labour is the design. Backends, jobs, queues and
        vendor cost belong to quantum-verifier; duplicating that surface
        would make two projects that are each worse than one."""
        forbidden = ("submit", "job", "backend", "queue", "provider", "account")
        for name in TOOLS:
            self.assertFalse(any(word in name for word in forbidden), name)


class RefusalsSurviveTheBoundaryTest(unittest.TestCase):

    def test_feasibility_refuses_a_circuit_with_no_signal(self):
        result = check_feasibility(5898, 18, 0.00311, 0.0293)
        self.assertFalse(result["mitigable"])
        self.assertLess(result["survival"], 1e-7)
        self.assertIn("compilation problem", result["verdict"])

    def test_recommendation_disqualifies_the_method_that_does_not_read_data(self):
        result = recommend_mitigation([
            {"name": "fraud", "errors": [0.1, 0.11, 0.09, 0.1],
             "cost": 1, "sensitivity": 0.02},
            {"name": "real", "errors": [0.4, 0.42, 0.38, 0.41],
             "cost": 5, "sensitivity": 1.05},
        ])
        self.assertEqual(result["recommended"], "real")
        self.assertEqual(result["disqualified"], ["fraud"])

    def test_recommendation_refuses_when_nothing_survives_the_circuit(self):
        result = recommend_mitigation(
            [{"name": "any", "errors": [1.0, 1.1, 0.9, 1.0], "sensitivity": 1.0}],
            two_qubit_gates=5898, n_qubits=18,
            two_qubit_error=0.00311, readout_error=0.0293)
        self.assertIsNone(result["recommended"])

    def test_a_tie_is_returned_as_a_tier_not_an_order(self):
        result = compare_two_methods("a", [1.15, 2.5, 0.1, 0.9],
                                     "b", [1.29, 2.1, 0.4, 1.2])
        self.assertFalse(result["distinguishable"])
        self.assertIsNone(result["better"])
        self.assertGreater(result["runs_each_to_settle"], 4)

    def test_falsify_reports_the_effect_against_its_own_bar(self):
        result = falsify_claim({"000": 3700, "111": 396},
                               {"000": 1873, "111": 2223},
                               marked=["000"], entangling_gates_removed=2)
        self.assertTrue(result["distinguishable_from_zero"])
        self.assertGreater(result["isolated_effect"], 0.4)

    def test_falsify_in_discovery_mode_quotes_the_null(self):
        import random
        rng = random.Random(4)
        def uniform(seed):
            gen = random.Random(seed)
            table = {}
            for _ in range(2000):
                key = format(gen.randrange(512), "09b")
                table[key] = table.get(key, 0) + 1
            return table
        result = falsify_claim(uniform(1), uniform(2))
        self.assertGreater(result["same_distribution_null"], 0.0,
                           "TVD between identical distributions is not zero")
        self.assertFalse(result["shift_established"])
        self.assertEqual(result["most_boosted"], [])

    def test_a_calibration_budget_refuses_to_quote_a_ceiling(self):
        result = budget_from_device(two_qubit_error=0.003, readout_error=0.03,
                                    measured_qubits=4, two_qubit_gates=20,
                                    shots=10_000)
        self.assertIn("not a quoted best case", result["note"])
        self.assertIn(result["dominant"], (None, "READOUT", "GATE_STOCHASTIC",
                                           "SHOT_NOISE"))


class QasmRefusalTest(unittest.TestCase):

    def test_mcx_and_mcz_are_refused_by_name_with_the_remedy(self):
        """The exact hour this project lost on the first outside circuit,
        returned as one line."""
        try:
            import qiskit  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("needs qiskit to parse")
        from qem_auditor.service import check_circuit_computes
        qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n'
                'mcx q[0],q[1],q[2],q[3],q[4];\n')
        with self.assertRaises(ValueError) as caught:
            check_circuit_computes(qasm, marked_inputs=[0], n_inputs=4)
        self.assertIn("not in qelib1.inc", str(caught.exception))
