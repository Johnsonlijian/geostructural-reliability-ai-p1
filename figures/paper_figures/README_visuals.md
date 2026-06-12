# Manuscript Figure Package

This folder contains deterministic, editable-first manuscript figure sources.

## Build

```powershell
python src/build_manuscript_figures.py
```

Outputs are written to `output/svg`, `output/pdf`, `output/png`, and `output/pptx`.
SVG/PDF plus scripts and source data are the canonical figure sources; PNG and PPTX files are previews.

## Style contract (publication style)

- Designed at 180 mm (7.05 in) print width; in-figure text 6-8 pt; panel letters bold lowercase.
- White background, thin axes (0.6 pt), outward ticks, no gridlines, frameless legends.
- No in-figure title banners, no rounded-card panels, no drop shadows, no gradients,
  no dashboard/infographic elements.
- Only Fig. 1 is a schematic/mechanism figure; Figs. 2-6 are data figures
  (Fig. 2a, Fig. 4a, and Fig. 5a contain small flat line-schematic panels only).
- Fig. 6 is an R16/R20 candidate closure figure generated for internal review; it is
  not referenced by the current five-figure manuscript unless the text is revised.
- Okabe-Ito colorblind-safe palette: SPT #0072B2, CPT #CC79A7, stable/excluded #009E73,
  critical/caveat #E69F00, failed #D55E00.

## Editing

- Edit SVG files in Inkscape or Illustrator.
- Use the PPTX files only for presentation review; they embed PNG previews and are not fully editable vector artwork.
- Do not redraw quantitative panels in PowerPoint.
- Keep `figure_provenance.yml` synchronized after manual edits.

## Policy Boundary

No generative-AI image model, stock image, web icon, or copyrighted visual asset is used.
All geometry is script-defined and all numerical values come from the manuscript/SI records mirrored in `data/`.
