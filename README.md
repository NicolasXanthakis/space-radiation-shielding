# Space Radiation Shielding

OpenGATE/Geant4 simulations of radiation shielding
and energy deposition in satellite electronics.

## Radiation Types

### Protons

Energetic protons occur in trapped radiation belts,
solar particle events and galactic cosmic rays.

A parallel beam with a single energy provides a controlled
first test of how proton energy and shielding thickness
affect dose in silicon.

Later models can include energy spectra, multiple directions
and other particle types to represent a specific space environment.

## Setup 1: radiation shielding for silicon chips

### First model

A parallel beam of 100 MeV protons passes through an Al
shield towards a silicon chip in vacuum.

Al thicknesses is compared while keeping the beam and
chip position fixed. Deposited energy, including secondary
particles, is converted to average absorbed dose: D = E / m.

Chip dose per source proton initially rises as protons slow
down, then drops sharply when the shield stops them.
A finer thickness sweep explores this transition.

This simplified model studies shielding effects; it does not
represent a complete orbital radiation environment or predict
electronic failures.

## Setup 2: radiation shielding for astronauts

### First model

For the simulated 100 MeV proton beam, thicker PVA hydrogel reduces the average tissue dose, but shifts the Bragg peak toward the tissue surface.
- From 0 to 60 mm shielding, the peak moves from 76.5 to 14.5 mm inside tissue, while its height remains roughly unchanged.
- At around 80 mm and above, recorded tissue dose becomes very small, consistent with primary protons stopping in the shield.


