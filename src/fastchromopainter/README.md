# fastchromopainter

Author: Prof. Simon Myers (myers@stats.ox.ac.uk)

## What this does
`fastchromopainter` is a fast C++ chromosome-painting implementation used to infer donor ancestry along haplotypes.  
Within this repository, it is used as an external/local-ancestry preprocessing component for downstream GhostCatcher analyses.

## Requirements
- GCC (or compatible `g++`)

## Compile
From `src/fastchromopainter`:

```bash
g++ -O3 -std=c++11 fastchromopainter.cpp -o fastchromopainter
```

This produces the executable `src/fastchromopainter/fastchromopainter`.

## Usage
Main painting mode:

```bash
./fastchromopainter ind_data_file panel_file donor_ids panel_labels rates_file output_file theta rho
```

Example (sanitized, relative paths):

```bash
chr=22
./src/fastchromopainter/fastchromopainter \
  data/target_chr${chr}.phase \
  data/panel_chr${chr}.binphase \
  data/donor.ids \
  data/donor.labels \
  data/chr${chr}.recombfile \
  output/chr${chr}_paint.out \
  0.0011 \
  368.43
```

Notes:
- `theta` is the mutation probability parameter.
- `rho` is the recombination scaling parameter.
- Input files must be aligned to the same chromosome/sites expected by the tool.
