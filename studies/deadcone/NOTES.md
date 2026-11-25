# Dead cone study: results

Status 2026-11-19. Full statistics, with the selections, settings and commands in [README.md](README.md).
All results are at detector level. Plots are in `output/results/` and `output/results_displaced/`.

## Samples and tagging

- Input: `/HEP/data/share/aleph/ntuples-withksloose/eventlevel`: 70 simulation files (qq̄) and 189 data files (1994).
- After the selection: 1.16M simulated jets and 1.91M data jets.
  The data/simulation ratio is 1.66, the same for all jets and for both tagged samples.
- Tagging the opposite jet:

  | Tag | Jets (sim / data) | Efficiency for b / c / uds events | Purity b / c / uds |
  |---|---|---|---|
  | b-tag (≥ 3 displaced tracks) | 79k / 127k | 28% / 2.5% / 0.4% | 91% / 6% / 3% |
  | light tag (no displaced tracks) | 738k / 1.21M | 24% / 59% / 80% | 8% / 16% / 76% |

  With the tighter displaced-track definition (Sip2dSig > 5, `settings_tight.json`), the tags change as follows:
  - b-tag: 44k / 71k jets, efficiency 16% / 0.8% / 0.1%, purity 94% / 4% / 2%;
  - light tag: 872k / 1.43M jets, efficiency 39% / 75% / 89%, purity 11% / 17% / 72%.

## Momentum space (ξ_p spectra)

Plots: `xi_charged.png` (all charged constituents) and `xi_prompt.png` (displaced tracks removed).

- **All charged constituents.**
  - The tag-corrected b and uds spectra in data agree with simulation and with OPAL in shape and normalization.
  - The b/uds ratio is the same in simulation, data and OPAL within a few percent over the full range:

    | ξ_p | 0.1 | 0.5 | 0.9 | 1.3 | 1.7 | 2.1 | 2.5 | 2.9 | 3.7 | 4.5 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | Simulation | 0.10 | 0.21 | 0.48 | 0.69 | 0.93 | 1.15 | 1.29 | 1.30 | 1.26 | 1.17 |
    | Data (tag-corrected) | 0.11 | 0.23 | 0.48 | 0.67 | 0.95 | 1.17 | 1.30 | 1.29 | 1.25 | 1.17 |
    | OPAL | 0.14 | 0.24 | 0.47 | 0.65 | 0.92 | 1.14 | 1.33 | 1.33 | 1.27 | 1.12 |

    Statistical uncertainties on the data ratio are about 0.01, except 0.03 in the first bin.
  - Charged multiplicity per event:
    - data: 18.9 (uds) and 22.2 (b);
    - OPAL (hadron level): 19.6 and 22.6.
- **Displaced tracks removed.**
  - The b and uds multiplicities become almost equal: 18.4 and 18.7 per event.
  - b/uds at intermediate momenta goes down: 0.37 at ξ_p = 0.9, 0.67 at 1.7 and 0.98 at 2.5, with data equal to simulation.
  - At the highest momenta (ξ_p < 0.6) it is unchanged at about 0.1–0.2.
  - The removal is only partial: decay tracks with small impact parameter remain, and prompt tracks with large impact parameter are lost.
    It is therefore not equivalent to the decay-product subtraction of Kluth et al.,
    who find the b/uds ratio of prompt particles suppressed to about 1/10 at high momentum.
  - The result depends on the displaced-track definition. With Sip2dSig > 5, fewer tracks are removed:
    - b/uds is 0.38 at ξ_p = 0.9, 0.71 at 1.7 and 1.06 at 2.5;
    - the multiplicities are 19.8 (b) and 18.6 (uds) per event.
- **Effect of the removal** (`xi_double_ratio.png`):
  - The double ratio (b/uds without displaced tracks) / (b/uds all charged) equals the fraction of tracks kept in b events
    relative to uds events. It shows where in ξ_p the removed tracks are and how much the removal changes b/uds.
    It measures the size of the decay-product correction, not the dead cone itself.
  - In b events, the fraction of tracks kept is lowest, 0.68, at ξ_p ≈ 1.7 (p ≈ 8 GeV). In uds events it is 0.97–1.0.
    The double ratio has its minimum of 0.71 there, and data and simulation agree within uncertainties in all bins.
  - At the highest momenta (ξ_p < 0.6) about 90% of the tracks in b events are kept, so the removal is least effective
    exactly where the dead cone suppression is expected.
  - In total, the removal takes out 3.5 tracks per b event and 0.5 per uds event. There are about 11
    charged B-decay products per b event (Kluth et al.), so the subtraction is far from complete.
  - With Sip2dSig > 5, the minimum of the double ratio is 0.76.
- **Robustness of the tag correction.** With the tighter definition, the b-tagged sample is purer but smaller.
  The tag-corrected b/uds ratio for all charged constituents is unchanged, within 0.03.

## Iterative declustering

Plots: `lund_ratios.png`, `lund_variations.png` and `lund_scaling.png`.
R(θ) = b/uds, normalized per radiator.

- **Data and simulation agree.** This holds in all E_rad bins, with and without the heavy-hadron proxy,
  except at the edge of phase space, where there are few entries.
- **At large angles, R ≈ 1** for E_rad ≥ 15 GeV. At lower E_rad, b radiators have fewer large-angle splittings (R ≈ 0.6–0.8).
- **R drops at small angles, independently of the splitting cut.**
  - Without any cut, or with z > 0.1, R levels off at the smallest angles.
  - kT cuts (0.2, 0.5 and 1 GeV) also remove the smallest angles, more so in b jets.
  - Higher kT cuts enhance a peak at ln(1/θ) ≈ 1.5 (R up to ~1.9). These are probably B decay products that were not merged into the proxy.
- **The energy dependence is opposite to the dead cone prediction.**

  | E_rad (GeV) | 10–15 | 15–20 | 20–30 | 30–50 |
  |---|---|---|---|---|
  | R < 1 from ln(1/θ) = (no cut; sim and data) | 2.6 | 2.4 | 2.1 | 1.9 |
  | Dead cone onset ln(E/m_b), m_b = 4.8 GeV | 1.0 | 1.3 | 1.7 | 2.1 |
  | R at ln(1/θ) ≈ 3.4–3.9 (no cut; sim) | 0.75 | 0.58 | 0.44 | 0.41 |

  - The drop sets in at *larger* angles for higher E_rad, and the suppression gets *stronger*.
    A dead cone predicts smaller angles and a weaker suppression at higher energy.
  - The curves for different E_rad do not coincide as a function of θ/θ₀.
  - As a function of kT, they roughly coincide for kT ≲ 1 GeV (E_rad ≥ 10 GeV).
  - c/uds shows the same pattern, weaker and at smaller angles.
- **Without the proxy** (`nomerge`), the declustering follows the harder branch.
  - b jets then show a large excess of splittings at ln(1/θ) ≈ 1.5–2 (R up to ~1.7 with the kT cut).
    These are B decay products filling the dead cone.
  - At high E_rad, the small-angle suppression is stronger than with the proxy.
- **Tighter displaced-track definition** (Sip2dSig > 5; `output/results_displaced/`):
  - The onset positions in the table above are unchanged, and data still agree with simulation.
  - The small-angle suppression at high E_rad is somewhat stronger: without a cut, R at ln(1/θ) ≈ 3.4–3.9 is
    0.75, 0.51, 0.34 and 0.28 for the four bins.
  - So the reversed energy dependence does not depend on the displaced-track definition.

## Interpretation

- **Momentum space.** The suppression of hard particles in b jets is reproduced in data and agrees with OPAL.
  To interpret it as a dead cone, decay products must be subtracted.
  The displaced-track removal does part of this; a full subtraction needs simulation, as in Kluth et al.
  This is the more promising observable for b quarks at LEP.
- **Declustering.** The data do not show the angular scaling expected from a dead cone.
  - At LEP, θ₀ = m_b/E_rad ≈ 0.1–0.5 rad, so emissions inside the b dead cone have kT ≲ z·m_b ≈ 0.5 GeV (z ~ 0.1).
    That is the hadronization regime.
  - R(θ) is then likely dominated by hard b fragmentation (the leading-particle effect):
    the B hadron takes most of the energy, which leaves few hard partners close to it.
  - The c quark (θ₀ ≈ 0.03–0.2 rad) may be more favourable, but needs a charm tag.

## Caveats and next steps

- **No simulation without the dead cone.** A generator-level Pythia study, with and without quark masses in the shower,
  would show what both observables should look like at LEP and how they respond to the dead cone.
- **Decay-product removal is partial.** It is based on impact parameters, and there is no truth information in the ntuples.
- **Simple tags.** Classifier scores via `--external_variables` would give purer and larger samples.
- **Detector level only.** Comparisons to MLLA predictions (Kluth et al.) need at least an efficiency and acceptance
  correction, and systematic uncertainties.
- **Correlated statistical uncertainties.** In the tag correction, the tagged simulation is a subset of the full simulation,
  and this correlation is ignored. The effect is small, since data statistics dominate.
