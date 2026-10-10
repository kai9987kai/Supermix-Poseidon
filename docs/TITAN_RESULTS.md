# TITAN matched-seed pilot results

This report describes the replay-verified local pilot at `outputs/titan_experiments/RECEIPT.json`. The result is a small synthetic-environment measurement, not evidence that TITAN is a better general-purpose controller.

## Configuration and integrity

| Setting | Value |
|---|---|
| Profile | `titan-six-arm-paired-replay-v2` |
| Environment | TidePool `tidepool-v1` |
| Scarcity | 2.5 |
| Horizon | 64 steps per episode |
| Seeds | 99100000–99100003 |
| Arms | Core, AURA, METAMORPH, CHIMERA, HYPERION, TITAN |
| TITAN decisions | 211 across four episodes |
| Receipt SHA-256 | `febf8d51bf8a68302c033c13d2743707a55e74392eff5f15aca2618ef65fe9b9` |
| Tidal checkpoint SHA-256 | `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27` |
| TidePool source SHA-256 | `f64b06b9434827b8ad6ce84212d8a7c62648abefa70661283de8b281c0acdbee` |

All six controllers were run once per shared seed. The receipt records each action, observation, returned reward, next observation, world state, and final snapshot. Verification rebuilt each controller and replayed the policy decisions and TidePool transitions. The CLI completed verification with `verified: true`, scope `receipt integrity, controller replay, and TidePool transition replay`; `promotion` is false.

## Descriptive outcomes

| Controller | Mean reward | Population SD | Survival | Mean steps |
|---|---:|---:|---:|---:|
| Frozen Tidal core | 6.4976 | 0.1388 | 100% (4/4) | 64.00 |
| AURA | 5.8650 | 0.3662 | 100% (4/4) | 64.00 |
| METAMORPH | 3.6554 | 4.3337 | 75% (3/4) | 52.00 |
| CHIMERA | 3.6710 | 4.3410 | 75% (3/4) | 52.00 |
| HYPERION | 3.4191 | 4.2182 | 75% (3/4) | 52.00 |
| TITAN | 3.4997 | 4.0795 | 75% (3/4) | 52.75 |

The paired mean TITAN reward deltas were −2.9979 versus the core, −2.3653 versus AURA, −0.1557 versus METAMORPH, −0.1713 versus CHIMERA, and +0.0806 versus HYPERION. TITAN tied HYPERION on three seeds and exceeded it on one; the small positive mean comes from running three extra steps on the seed where both controllers died. TITAN did not survive all episodes, and the core and AURA had higher mean reward and survival in this run. These values do not support a superiority claim.

For seed 99100003, TITAN died from dehydration at step 19. METAMORPH, CHIMERA, and HYPERION died from dehydration plus exhaustion at step 16, while the core and AURA reached the 64-step horizon. This is one paired scenario and needs broader, preregistered evaluation before any design conclusion.

## TITAN telemetry

Recorded means were spectral divergence 0.058570, synthetic amplitude entropy 0.454479, lattice coherence 0.726326, composite coherence 0.609700, classical circuit fidelity proxy 1.000000, simulated buoyancy force 0.509943, and toy trophic richness 0.401843. These are implementation telemetry fields. In particular, “fidelity,” “quantum,” “buoyancy,” and “trophic” do not refer to quantum hardware, real fluids, or a biological ecosystem.

## Reproduce and verify

```powershell
python -m poseidon titan-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 99100000
python -m poseidon verify-titan outputs/titan_experiments/RECEIPT.json
```

The seed range, active Tidal checkpoint, and TidePool source must match this receipt to reproduce its exact digest. The benchmark does not train weights, create a fitted artifact, or activate TITAN as the default policy. See [design and limitations](TITAN_DESIGN.md).
