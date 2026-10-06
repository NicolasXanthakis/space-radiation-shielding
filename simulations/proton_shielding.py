from pathlib import Path

import itk
import numpy as np
import opengate as gate
from opengate.utility import g4_units


def run_sim():
    mm = g4_units.mm
    MeV = g4_units.MeV

    n_protons = 10_000
    energy_MeV = 100
    shield_mm = 2
    gap_mm = 5
    chip_thickness_mm = 0.5

    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.random_seed = 42
    sim.visu = False

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(exist_ok=True)
    sim.output_dir = output_dir

    # Vacuum world
    sim.world.size = [200 * mm] * 3
    sim.world.material = "G4_Galactic"

    # Aluminium shield centred at Z = 0
    shield = sim.add_volume("Box", "shield")
    shield.size = [50 * mm, 50 * mm, shield_mm * mm]
    shield.material = "G4_Al"
    shield.color = [0.7, 0.7, 0.7, 0.5]

    # Silicon chip behind the shield, with a 5 mm surface gap
    chip = sim.add_volume("Box", "chip")
    chip.size = [10 * mm, 10 * mm, chip_thickness_mm * mm]
    chip.translation = [
        0,
        0,
        (shield_mm / 2 + gap_mm + chip_thickness_mm / 2) * mm,
    ]
    chip.material = "G4_Si"
    chip.color = [0.2, 0.7, 0.2, 1.0]

    # Parallel proton beam, smaller than the chip
    source = sim.add_source("GenericSource", "protons")
    source.particle = "proton"
    source.n = n_protons
    source.position.type = "disc"
    source.position.radius = 4 * mm
    source.position.translation = [0, 0, -50 * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.energy.type = "mono"
    source.energy.mono = energy_MeV * MeV

    # Include electromagnetic and nuclear interactions
    sim.physics_manager.physics_list_name = "FTFP_BERT"

    # Explicit secondary production cuts
    for volume in (shield, chip):
        sim.physics_manager.set_production_cut(
            volume.name, "all", 0.01 * mm
        )

    # One scoring voxel covering the entire chip
    scorer = sim.add_actor("DoseActor", "chip_edep")
    scorer.attached_to = chip.name
    scorer.size = [1, 1, 1]
    scorer.spacing = chip.size
    scorer.edep.write_to_disk = False

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True

    sim.run()
    print(stats)

    # Total deposited energy, including secondary particles
    image = scorer.edep.get_data()
    total_MeV = float(np.sum(itk.array_from_image(image)) / MeV)

    # G4_Si density: 2.33 g/cm³
    chip_volume_cm3 = float(np.prod(np.array(chip.size) / mm)) / 1000
    chip_mass_kg = 2.33 * chip_volume_cm3 / 1000

    # Convert MeV to joules; dose = deposited energy / mass
    dose_Gy = total_MeV * 1.602176634e-13 / chip_mass_kg

    print(f"\nProton energy: {energy_MeV} MeV")
    print(f"Aluminium thickness: {shield_mm} mm")
    print(f"Chip mass: {chip_mass_kg * 1000:.4f} g")
    print(f"Deposited energy: {total_MeV:.3f} MeV")
    print(f"Mean deposited energy/source proton: "
          f"{total_MeV / n_protons:.5f} MeV")
    print(f"Mean chip dose for {n_protons} source protons: "
          f"{dose_Gy:.6e} Gy")
    print(f"Dose/source proton: {dose_Gy / n_protons:.6e} Gy")


if __name__ == "__main__":
    run_sim()