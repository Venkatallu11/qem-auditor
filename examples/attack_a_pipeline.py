#!/usr/bin/env python3
"""Attacking a real fitting pipeline, and telling a genuine one from a
merely flexible one.

T_label, T_sign and T_shot become executable once a pipeline implements
the Reconstructor interface: fit, reconstruct, goodness_of_fit.

Two pipelines are defined below, and the whole point is that they look
equally good -- in fact the flexible one looks BETTER -- on the real data:

    ScaleReconstructor    a CDR-shaped correction. Fits one attenuation
                          factor per LABEL on training slots where the
                          ideal value is known, then applies it to the
                          target slot. It depends on which label a value
                          belongs to.

    FlexibleReconstructor one free parameter per measurement. It fits the
                          real data PERFECTLY (chi2/dof = 0) -- and it
                          fits shuffled or negated data just as perfectly,
                          because it has enough freedom to fit anything.

Judged on fit quality alone you would pick the flexible one. Only the
attacks tell them apart. That is the entire argument for running them.

Needs nothing beyond the standard library.
"""
import random
from statistics import mean

from qem_auditor.adversary import AttackPlan, GRAMMAR
from qem_auditor.executor import AttackExecutor
from qem_auditor.reconstruct import FitData, Measurement
from qem_auditor.schema import Controls, Experiment, Outputs

IDEAL = {"XX": 0.80, "YY": -0.60, "ZZ": 0.40}
ATTENUATION = {"XX": 0.90, "YY": 0.75, "ZZ": 0.85}
TRAINING_SLOTS = ["u0", "u1", "u2"]   # ideal known here
TARGET_SLOT = "u3"                     # the thing being measured


def make_data(draws: int = 4, noise: float = 0.02, seed: int = 0) -> FitData:
    rng = random.Random(seed)
    measurements = []
    for draw in range(draws):
        for slot in TRAINING_SLOTS + [TARGET_SLOT]:
            for label, ideal in IDEAL.items():
                value = ideal * ATTENUATION[label] + rng.gauss(0, noise)
                measurements.append(Measurement(slot=slot, label=label, value=value,
                                                sigma=noise, shots=20_000, draw=draw))
    return FitData(measurements, {"ideal": IDEAL})


class ScaleReconstructor:
    """Genuine: per-label attenuation fitted on training slots."""

    def fit(self, data: FitData) -> dict:
        scales = {}
        for label, ideal in IDEAL.items():
            vals = [m.value for m in data.measurements
                    if m.label == label and m.slot in TRAINING_SLOTS]
            scales[label] = (mean(vals) / ideal) if vals and ideal else 1.0
        return scales

    def reconstruct(self, fit: dict, data: FitData) -> float:
        """Correct the TARGET slot using scales fitted elsewhere, so the
        answer genuinely depends on the data rather than being the ideal
        by construction."""
        total = 0.0
        for label in IDEAL:
            vals = [m.value for m in data.measurements
                    if m.label == label and m.slot == TARGET_SLOT]
            scale = fit.get(label) or 1.0
            if vals and abs(scale) > 1e-9:
                total += mean(vals) / scale
        return total

    def goodness_of_fit(self, fit: dict, data: FitData) -> float:
        residuals = []
        for m in data.measurements:
            if m.slot not in TRAINING_SLOTS:
                continue
            predicted = IDEAL.get(m.label, 0.0) * fit.get(m.label, 1.0)
            sigma = m.sigma or 1e-6
            residuals.append(((m.value - predicted) / sigma) ** 2)
        return sum(residuals) / max(1, len(residuals))


class FullyInstrumentedReconstructor(ScaleReconstructor):
    """The genuine pipeline, plus the three optional capabilities that make
    the rest of the grammar executable.

    Nothing here is exotic: a method that has free parameters can expose
    them, a method that assumes a noise model can expose its intervals,
    and a method that imposes a structural constraint can fit without it.
    Each is a handful of lines, and each unlocks an attack that otherwise
    has to be taken on trust.
    """

    # -- Parameterised: T_leakage ------------------------------------
    def free_parameters(self):
        # (floor, nominal). The floor is the limit worth probing -- for a
        # CDR-style method, a training radius shrinking to zero.
        return {"training_radius": (0.0, 1.0)}

    def evaluate_at(self, name, value, data):
        """At radius 1.0 the correction is fitted from the data. As the
        radius goes to zero this (deliberately degenerate) variant slides
        toward returning the known ideal regardless of what was measured
        -- which is exactly the disqualified CDR behaviour."""
        fitted = self.reconstruct(self.fit(data), data)
        known_answer = sum(IDEAL.values())
        return value * fitted + (1.0 - value) * known_answer

    # -- NoiseModelled: T_calibration --------------------------------
    def noise_parameters(self):
        return {label: (att * 0.97, att * 1.03) for label, att in ATTENUATION.items()}

    def evaluate_under_noise(self, params, data):
        total = 0.0
        for label in IDEAL:
            vals = [m.value for m in data.measurements
                    if m.label == label and m.slot == TARGET_SLOT]
            scale = params.get(label) or 1.0
            if vals:
                total += mean(vals) / scale
        return total

    # -- StructurallyConstrained: T_correlation ----------------------
    def fit_without_structure(self, data):
        """Drop the per-label pooling: one scale per (slot, label), fitted
        from a single measurement each. Identifiable in principle, far
        noisier in practice."""
        out = {}
        for m in data.measurements:
            if m.slot in TRAINING_SLOTS and IDEAL.get(m.label):
                out[(m.slot, m.label)] = m.value / IDEAL[m.label]
        return out

    def reconstruct_without_structure(self, fit, data):
        total = 0.0
        for label in IDEAL:
            scales = [v for (slot, lab), v in fit.items() if lab == label]
            vals = [m.value for m in data.measurements
                    if m.label == label and m.slot == TARGET_SLOT]
            if scales and vals and abs(scales[0]) > 1e-9:
                total += mean(vals) / scales[0]
        return total


class FlexibleReconstructor:
    """One parameter per measurement. Not a strawman -- this is what an
    over-parameterised correction looks like, and on real data it reports
    a better chi2/dof than the genuine model."""

    def fit(self, data: FitData) -> dict:
        return {(m.slot, m.label, m.draw): m.value for m in data.measurements}

    def reconstruct(self, fit: dict, data: FitData) -> float:
        total = 0.0
        for label in IDEAL:
            vals = [v for (slot, lab, _), v in fit.items()
                    if lab == label and slot == TARGET_SLOT]
            if vals:
                total += mean(vals) / ATTENUATION[label]
        return total

    def goodness_of_fit(self, fit: dict, data: FitData) -> float:
        residuals = []
        for m in data.measurements:
            predicted = fit.get((m.slot, m.label, m.draw), m.value)
            sigma = m.sigma or 1e-6
            residuals.append(((m.value - predicted) / sigma) ** 2)
        return sum(residuals) / max(1, len(residuals))


def main() -> None:
    data = make_data()
    exp = Experiment(
        experiment_id="pipeline_under_attack",
        claim="This correction recovers the ideal expectation values.",
        description="A per-label correction fitted to real measurements.",
        backend="local", shots=20_000, controls=Controls(), outputs=Outputs())
    attacks = [GRAMMAR[n](exp) for n in ("T_label", "T_sign", "T_shot")]
    truth = sum(IDEAL.values())

    for name, reconstructor in (("GENUINE  (per-label scale, CDR-shaped)",
                                 ScaleReconstructor()),
                                ("FLEXIBLE (one parameter per measurement)",
                                 FlexibleReconstructor())):
        fit = reconstructor.fit(data)
        print("=" * 74)
        print(name)
        print("=" * 74)
        print(f"  chi2/dof on the REAL data : "
              f"{reconstructor.goodness_of_fit(fit, data):.6g}")
        print(f"  reconstructed             : "
              f"{reconstructor.reconstruct(fit, data):.4f}   (truth {truth:.4f})")
        print("  -> on fit quality alone, the flexible model looks better.")
        print()

        report = AttackExecutor().run(
            exp, AttackPlan(exp.experiment_id, list(attacks)),
            reconstructor=reconstructor, fit_data=data)
        for outcome in report.outcomes:
            verdict = ("SURVIVED" if outcome.survived else
                       "FALSIFIED" if outcome.survived is False else "NOT RUN")
            print(f"  [{verdict:9}] {outcome.attack.attack_id}")
            print(f"              {outcome.detail}")
        print()

    # The three that need the optional capabilities.
    print("=" * 74)
    print("WITH THE OPTIONAL CAPABILITIES IMPLEMENTED")
    print("=" * 74)
    print("  free_parameters/evaluate_at, noise_parameters/evaluate_under_noise,")
    print("  and fit_without_structure -- a handful of lines each.")
    print()
    instrumented = FullyInstrumentedReconstructor()
    extra = [GRAMMAR[n](exp) for n in ("T_leakage", "T_calibration", "T_correlation")]
    report = AttackExecutor().run(
        exp, AttackPlan(exp.experiment_id, extra),
        reconstructor=instrumented, fit_data=data)
    for outcome in report.outcomes:
        verdict = ("SURVIVED" if outcome.survived else
                   "FALSIFIED" if outcome.survived is False else "NOT RUN")
        print(f"  [{verdict:9}] {outcome.attack.attack_id}")
        print(f"              {outcome.detail}")
    print()

    print("=" * 74)
    print("READING THIS")
    print("=" * 74)
    print("""
  T_label separates them, which is the point. The genuine model's fit gets
  ~500x worse when the label correspondence is destroyed; the flexible one
  does not notice at all, because it never used the correspondence. Yet the
  flexible model has the BETTER chi2/dof on real data. Fit quality alone
  would have picked the wrong one.

  T_sign falsifies BOTH, and that is a real finding rather than a false
  alarm. A per-label scale correction is genuinely sign-agnostic: negating
  every measurement just flips the fitted scale, and the fit is equally
  good. So a sign or bookkeeping error anywhere upstream would not show up
  in this model's goodness-of-fit -- worth knowing about a method you were
  about to trust.

  T_shot reports shot noise dominating here, which is correct for this toy:
  there is no expensive per-draw Monte Carlo in it. On the real H4 pipeline
  the same test returns the opposite, and that is the case where "run more
  shots" is the expensive wrong answer.

  T_leakage catches the degenerate variant. As the training radius goes to
  zero, that pipeline slides toward returning the known answer whatever was
  measured -- so the gap between its output on real and on SCRAMBLED data
  collapses. Note the statistic: dependence on the data, not distance to
  the truth. An auditor holding the true answer would not need to be an
  auditor.
""")


if __name__ == "__main__":
    main()
