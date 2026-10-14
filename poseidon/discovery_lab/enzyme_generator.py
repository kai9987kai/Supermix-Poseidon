"""Project PROMETHEUS Discovery Lab: Multimodal Biocatalytic Enzyme Generator.

Implements generative catalytic enzyme synthesis bridging multimodal 3D geometric pockets,
amino acid motif generation, transition-state barrier reduction (Delta Delta G^#),
and Michaelis-Menten kinetics for structural biopolymer cross-linking (e.g., binding logs into rafts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# Standard 20 amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Catalytic motifs associated with lignin cross-linking, transglycosylation, and plant fiber polymerization
CATALYTIC_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "laccase_polyphenol_oxidase": {
        "triad": ["His-64", "Cys-85", "His-102"],
        "cofactor": "Cu2+",
        "target_substrate": "lignocellulose",
        "uncatalyzed_barrier_kcal": 28.5,
        "k_cat_base": 145.0,  # s^-1
        "km_base": 0.35,      # mM
        "bonding_strength": 0.95,
        "water_resistance": 0.98,
    },
    "transglycosylase_cellulose_synthase": {
        "triad": ["Asp-142", "Glu-228", "Asp-310"],
        "cofactor": "Mg2+",
        "target_substrate": "hemicellulose",
        "uncatalyzed_barrier_kcal": 24.0,
        "k_cat_base": 88.0,
        "km_base": 0.50,
        "bonding_strength": 0.91,
        "water_resistance": 0.94,
    },
    "peroxidase_bioadhesive": {
        "triad": ["Arg-38", "His-42", "Phe-172"],
        "cofactor": "Heme-Fe3+",
        "target_substrate": "tannin_polyphenol",
        "uncatalyzed_barrier_kcal": 31.0,
        "k_cat_base": 210.0,
        "km_base": 0.20,
        "bonding_strength": 0.99,
        "water_resistance": 0.99,
    },
}


@dataclass
class CatalyticActiveSite:
    """3D coordinates and amino acid residue assignments for catalytic binding pocket."""
    residues: List[str]
    pocket_coordinates_3d: List[List[float]]  # [x, y, z] for each catalytic residue
    pocket_volume_angstrom3: float
    surface_hydrophobicity: float  # -1.0 (hydrophilic) to +1.0 (hydrophobic)
    charge_density: float


@dataclass
class SynthesizedEnzyme:
    """Complete specification of a generated catalytic enzyme."""
    enzyme_id: str
    name: str
    target_class: str
    sequence_fragment: str
    active_site: CatalyticActiveSite
    v_max: float               # umol / (min * mg)
    k_m: float                 # mM (Michaelis constant)
    k_cat: float               # s^-1 (turnover number)
    delta_delta_g_barrier: float  # kcal/mol reduction in activation energy
    cure_time_ticks: int       # simulation steps required for full polymerization
    cured_bond_strength: float # 0.0 to 1.0 tensile bond strength
    water_proof_rating: float  # 0.0 to 1.0 stability in turbulent water
    synthesis_receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enzyme_id": self.enzyme_id,
            "name": self.name,
            "target_class": self.target_class,
            "sequence_fragment": self.sequence_fragment,
            "active_site": {
                "residues": self.active_site.residues,
                "pocket_coordinates_3d": self.active_site.pocket_coordinates_3d,
                "pocket_volume_angstrom3": round(float(self.active_site.pocket_volume_angstrom3), 2),
                "surface_hydrophobicity": round(float(self.active_site.surface_hydrophobicity), 3),
                "charge_density": round(float(self.active_site.charge_density), 3),
            },
            "v_max": round(float(self.v_max), 2),
            "k_m": round(float(self.k_m), 4),
            "k_cat": round(float(self.k_cat), 2),
            "delta_delta_g_barrier_kcal": round(float(self.delta_delta_g_barrier), 2),
            "cure_time_ticks": int(self.cure_time_ticks),
            "cured_bond_strength": round(float(self.cured_bond_strength), 3),
            "water_proof_rating": round(float(self.water_proof_rating), 3),
            "synthesis_receipt_hash": self.synthesis_receipt_hash,
        }


class MultimodalEnzymeGenerator:
    """Generates specialized catalytic enzymes tailored to environmental substrates & physical bonding demands."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.synthesis_history: List[SynthesizedEnzyme] = []

    def generate_catalyst_for_substrate(
        self,
        substrate_type: str = "lignocellulose",
        target_shear_force: float = 250.0,
        ambient_temperature_c: float = 22.0,
        water_exposure: float = 0.85,
    ) -> SynthesizedEnzyme:
        """Synthesizes an optimal enzyme catalyst lowering delta G^# for structural bio-polymerization."""
        # Select matching catalytic archetype
        if "ligno" in substrate_type or "wood" in substrate_type or "log" in substrate_type:
            archetype_key = "laccase_polyphenol_oxidase"
        elif "reed" in substrate_type or "grass" in substrate_type or "cellul" in substrate_type:
            archetype_key = "transglycosylase_cellulose_synthase"
        else:
            archetype_key = "peroxidase_bioadhesive"

        spec = CATALYTIC_SIGNATURES[archetype_key]

        # 1. Generate active pocket 3D geometry
        num_residues = len(spec["triad"])
        pocket_coords = []
        for i in range(num_residues):
            angle = (2.0 * math.pi * i) / num_residues
            radius = 3.2 + float(self.rng.uniform(-0.3, 0.3))
            z = float(self.rng.uniform(-1.0, 1.0))
            pocket_coords.append([round(radius * math.cos(angle), 3), round(radius * math.sin(angle), 3), round(z, 3)])

        pocket_vol = float(180.0 + self.rng.uniform(10.0, 50.0))
        hydrophobicity = float(0.45 + self.rng.uniform(-0.1, 0.2))
        charge = float(-1.2 + self.rng.uniform(-0.5, 0.5))

        active_site = CatalyticActiveSite(
            residues=list(spec["triad"]),
            pocket_coordinates_3d=pocket_coords,
            pocket_volume_angstrom3=pocket_vol,
            surface_hydrophobicity=hydrophobicity,
            charge_density=charge,
        )

        # 2. Synthesize pseudo-sequence fragment around triad
        seq_parts = []
        for res in spec["triad"]:
            flank_left = "".join(self.rng.choice(list(AMINO_ACIDS), size=4))
            flank_right = "".join(self.rng.choice(list(AMINO_ACIDS), size=4))
            single_letter = res.split("-")[0][:1]
            seq_parts.append(f"{flank_left}{single_letter}{flank_right}")
        sequence_fragment = "-".join(seq_parts)

        # 3. Calculate Michaelis-Menten Kinetics and Transition State Delta Delta G
        r_gas_const = 0.001987  # kcal / (mol * K)
        temp_k = ambient_temperature_c + 273.15
        k_cat = spec["k_cat_base"] * (1.0 + float(self.rng.uniform(-0.08, 0.15)))
        k_uncat = 1e-4  # baseline spontaneous cross-linking rate (s^-1)

        # Arrhenius transition-state barrier reduction
        delta_delta_g = float(-r_gas_const * temp_k * math.log(k_cat / k_uncat))
        v_max = float(k_cat * 0.85)
        k_m = float(spec["km_base"] * (1.0 + self.rng.uniform(-0.05, 0.05)))

        cured_strength = min(1.0, float(spec["bonding_strength"] * (1.0 + target_shear_force / 10000.0)))
        water_proof = min(1.0, float(spec["water_resistance"] * (1.0 - (1.0 - water_exposure) * 0.05)))
        cure_ticks = max(1, int(round(3.0 - (k_cat / 150.0))))

        # 4. Generate unique cryptographic synthesis hash
        raw_signature = f"{archetype_key}:{sequence_fragment}:{delta_delta_g:.3f}:{k_cat:.2f}"
        synthesis_hash = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()
        enzyme_id = f"ENZ-{synthesis_hash[:10].upper()}"

        enzyme = SynthesizedEnzyme(
            enzyme_id=enzyme_id,
            name=f"BioCatalyst-{archetype_key.split('_')[0].capitalize()}",
            target_class=archetype_key,
            sequence_fragment=sequence_fragment,
            active_site=active_site,
            v_max=v_max,
            k_m=k_m,
            k_cat=k_cat,
            delta_delta_g_barrier=delta_delta_g,
            cure_time_ticks=cure_ticks,
            cured_bond_strength=cured_strength,
            water_proof_rating=water_proof,
            synthesis_receipt_hash=synthesis_hash,
        )

        self.synthesis_history.append(enzyme)
        return enzyme
