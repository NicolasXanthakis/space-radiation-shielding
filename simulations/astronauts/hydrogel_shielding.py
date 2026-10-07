from pathlib import Path

import itk
import numpy as np
import opengate as gate
from opengate.utility import g4_units


def run_sim(shield_mm=0, n_protons=10_000, seed=1):
    mm = g4_units.mm
    MeV = g4_units.MeV

    energy_MeV = 100
    gap_mm = 5
    shield_back_z_mm = 1

    # PVA hydrogel: 90% water and 10% polyvinyl alcohol 
    shield_material = "PVA_Hydrogel"

    tissue_width_mm = 50
    tissue_depth_mm = 100
    slice_thickness_mm = 1
    n_slices = int(tissue_depth_mm / slice_thickness_mm)

    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.random_seed = seed
    sim.visu = False


    # Hydrogel
    # Water: H2O; PVA repeat unit: C2H4O.
    water_fraction = 0.90
    pva_fraction = 0.10

    # Atomic masses used to calculate elemental mass fractions.
    H = 1.008
    C = 12.011
    O = 15.999

    water_molar_mass = 2 * H + O
    pva_repeat_molar_mass = 2 * C + 4 * H + O

    hydrogen_fraction = (
        water_fraction * (2 * H / water_molar_mass)
        + pva_fraction * (4 * H / pva_repeat_molar_mass)
    )

    carbon_fraction = (
        pva_fraction * (2 * C / pva_repeat_molar_mass)
    )

    oxygen_fraction = (
        water_fraction * (O / water_molar_mass)
        + pva_fraction * (O / pva_repeat_molar_mass)
    )

    # Estimated mixture density assuming additive constituent volumes.
    # These are assumed constituent densities, not measured gel data.
    water_density_g_cm3 = 1.00
    pva_density_g_cm3 = 1.27

    hydrogel_density_g_cm3 = 1.0 / (
        water_fraction / water_density_g_cm3
        + pva_fraction / pva_density_g_cm3
    )

    sim.volume_manager.material_database.add_material_weights(
        shield_material,
        ["H", "C", "O"],
        [hydrogen_fraction, carbon_fraction, oxygen_fraction],
        hydrogel_density_g_cm3 * g4_units.g_cm3,
    )

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "hydrogel_tissue"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sim.output_dir = output_dir

    # Large enough for the shield, source and tissue.
    sim.world.size = [400 * mm] * 3
    sim.world.material = "G4_Galactic"

    # Keep the downstream shield surface fixed.
    shield = None
    if shield_mm > 0:
        shield = sim.add_volume("Box", "shield")
        shield.size = [100 * mm, 100 * mm, shield_mm * mm]
        shield.translation = [
            0,
            0,
            (shield_back_z_mm - shield_mm / 2) * mm,
        ]
        shield.material = shield_material
        shield.color = [0.2, 0.6, 1.0, 0.5]

    # Water phantom representing tissue.
    tissue_front_z_mm = shield_back_z_mm + gap_mm

    tissue = sim.add_volume("Box", "tissue")
    tissue.size = [
        tissue_width_mm * mm,
        tissue_width_mm * mm,
        tissue_depth_mm * mm,
    ]
    tissue.translation = [
        0,
        0,
        (tissue_front_z_mm + tissue_depth_mm / 2) * mm,
    ]
    tissue.material = "G4_WATER"
    tissue.color = [0.9, 0.5, 0.5, 0.5]

    # Fixed source upstream of every tested shield.
    source_z_mm = -150
    if shield_back_z_mm - shield_mm <= source_z_mm:
        raise ValueError("Shield reaches or extends beyond the source.")

    source = sim.add_source("GenericSource", "protons")
    source.particle = "proton"
    source.n = n_protons
    source.position.type = "disc"
    source.position.radius = 4 * mm
    source.position.translation = [0, 0, source_z_mm * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.energy.type = "mono"
    source.energy.mono = energy_MeV * MeV

    sim.physics_manager.physics_list_name = "FTFP_BERT"

    volumes = [tissue]
    if shield is not None:
        volumes.append(shield)

    for volume in volumes:
        sim.physics_manager.set_production_cut(
            volume.name, "all", 0.01 * mm
        )

    # One voxel across X/Y; 1 mm slices along Z.
    # Each slice dose is averaged over its full 50 × 50 mm area.
    scorer = sim.add_actor("DoseActor", "tissue_edep")
    scorer.attached_to = tissue.name
    scorer.size = [1, 1, n_slices]
    scorer.spacing = [
        tissue_width_mm * mm,
        tissue_width_mm * mm,
        slice_thickness_mm * mm,
    ]
    scorer.edep.write_to_disk = False

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True

    sim.run(start_new_process=True)
    print(stats)

    # ITK arrays use Z, Y, X ordering.
    image = scorer.edep.get_data()
    edep_array = np.asarray(
        itk.array_from_image(image), dtype=np.float64
    )
    slice_edep_MeV = edep_array.sum(axis=(1, 2)) / MeV

    # G4_WATER density: 1 g/cm³.
    density_g_cm3 = 1.0
    slice_volume_cm3 = (
        tissue_width_mm**2 * slice_thickness_mm / 1000
    )
    slice_mass_kg = density_g_cm3 * slice_volume_cm3 / 1000
    tissue_mass_kg = slice_mass_kg * n_slices

    MeV_to_J = 1.602176634e-13

    slice_dose_per_proton = (
        slice_edep_MeV * MeV_to_J
        / slice_mass_kg
        / n_protons
    )

    total_edep_MeV = float(slice_edep_MeV.sum())
    mean_dose_per_proton = (
        total_edep_MeV * MeV_to_J
        / tissue_mass_kg
        / n_protons
    )

    # Depth measured from the tissue entrance, at slice centres.
    depths_mm = (
        np.arange(n_slices) + 0.5
    ) * slice_thickness_mm

    peak_index = int(np.argmax(slice_dose_per_proton))

    np.savetxt(
        output_dir / f"depth_dose_{shield_mm:g}mm.csv",
        np.column_stack([depths_mm, slice_dose_per_proton]),
        delimiter=",",
        header="tissue_depth_mm,slice_dose_per_source_proton_Gy",
        comments="",
    )

    print(f"\nProton energy: {energy_MeV} MeV")

    print(f"PVA hydrogel thickness: {shield_mm:g} mm")
    print(f"Hydrogel density: {hydrogel_density_g_cm3:.4f} g/cm³")
    
    print(f"Tissue mass: {tissue_mass_kg * 1000:.2f} g")
    print(f"Deposited energy: {total_edep_MeV:.3f} MeV")
    print(
        "Mean tissue dose/source proton: "
        f"{mean_dose_per_proton:.6e} Gy"
    )

    if total_edep_MeV > 0:
        print(
            "Highest slice dose/source proton: "
            f"{slice_dose_per_proton[peak_index]:.6e} Gy "
            f"at {depths_mm[peak_index]:.1f} mm tissue depth"
        )
    else:
        print("No energy deposition recorded in tissue.")

    return mean_dose_per_proton


if __name__ == "__main__":
    thicknesses_mm = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    doses_per_proton = []

    for i, thickness in enumerate(thicknesses_mm):
        dose = run_sim(
            shield_mm=thickness,
            n_protons=10_000,
            seed=1 + i,
        )
        doses_per_proton.append(dose)

    print("\nShield thickness (mm) | Mean tissue dose/proton (Gy)")
    for thickness, dose in zip(thicknesses_mm, doses_per_proton):
        print(f"{thickness:21.1f} | {dose:.6e}")