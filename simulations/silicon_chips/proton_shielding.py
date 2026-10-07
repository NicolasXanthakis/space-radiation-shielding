from pathlib import Path

import itk
import numpy as np
import opengate as gate
from opengate.utility import g4_units


def run_sim(shield_mm=2, n_protons=10_000, seed=1):
    mm = g4_units.mm
    MeV = g4_units.MeV

    energy_MeV = 100
    gap_mm = 5
    chip_thickness_mm = 0.5

    shield_back_z_mm = 1 

    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.random_seed = seed
    sim.visu = False

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(exist_ok=True)
    sim.output_dir = output_dir

    # World (vacuum)
    sim.world.size = [200 * mm] * 3
    sim.world.material = "G4_Galactic"

    # Aluminium shield centred at Z = 0
    # Keep the downstream shield surface fixed; grow toward the source
    shield = None

    if shield_mm > 0:
        shield = sim.add_volume("Box", "shield")
        shield.size = [50 * mm, 50 * mm, shield_mm * mm]
        shield.translation = [
            0, 0, (shield_back_z_mm - shield_mm / 2) * mm
        ]
        shield.material = "G4_Al"
        shield.color = [0.7, 0.7, 0.7, 0.5]

    # Chip position stays fixed for every shield thickness
    chip = sim.add_volume("Box", "chip")
    chip.size = [10 * mm, 10 * mm, chip_thickness_mm * mm]
    chip.translation = [
        0,
        0,
        (shield_back_z_mm + gap_mm + chip_thickness_mm / 2) * mm,
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
    volumes = [chip]
    if shield is not None:
        volumes.append(shield)

    for volume in volumes:
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

    sim.run(start_new_process=True)
    print(stats)

    # Total E_Dep (including secondary particles)
    image = scorer.edep.get_data()
    total_MeV = float(np.sum(itk.array_from_image(image)) / MeV)

    # G4_Si density (2.33 g/cm3)
    chip_volume_cm3 = float(np.prod(np.array(chip.size) / mm)) / 1000
    chip_mass_kg = 2.33 * chip_volume_cm3 / 1000

    # Convert MeV to joules; dose = E_Dep/mass
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

    return dose_Gy / n_protons

if __name__ == "__main__":

    # thicknesses_mm = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    thicknesses_mm = [34, 34.5, 35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40]
    
    doses_per_proton = []

    for i, thickness in enumerate(thicknesses_mm):
        dose = run_sim(
            shield_mm=thickness,
            n_protons=10_000,
            seed=1 + i,
        )
        doses_per_proton.append(dose)

    print("\nAl thickness (mm) | Chip dose/source proton (Gy)")
    for thickness, dose in zip(thicknesses_mm, doses_per_proton):
        print(f"{thickness:17.1f} | {dose:.6e}")