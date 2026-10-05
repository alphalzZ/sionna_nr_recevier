---
name: blosm-sionna-rt-export
description: Convert Blosm/OpenStreetMap Blender scenes into this project's restricted Sionna RT scene bundle (scene.xml v2.1.0 plus ASCII PLY meshes) through Blender MCP, then validate and load it with Sionna RT.
---

# Blosm → Sionna RT scene export

Use this skill when a user asks to turn a Blosm/OpenStreetMap Blender scene into an importable scene for this repository's Sionna RT workflow.

## Inputs and safety

- Inspect the selected `.blend` scene with Blender MCP tools (`blender-scene`, geometry inspection); do not use shell/local Python to inspect or modify Blender data.
- Use direct Blender MCP calls from the current DCC instance. Do not use `omp --cwd` for Blender operations.
- Preserve the source `.blend` and OSM files. Open the source in Blender MCP and do not save it after conversion.
- Export to a new, separate scene root. Refuse to overwrite an existing output unless the user explicitly requests replacement.
- Prefer typed Blender MCP tools. Search for a typed PLY exporter first. If none is available, use `blender-dev__run_script` through Blender MCP as the last resort; keep the temporary script outside source assets and remove it after the export.
- Do not infer roads or other layers from OSM data that are absent from the opened Blender scene. Report omitted feature classes.

## Sionna RT package contract

The project accepts a deliberately limited Mitsuba XML/ASCII-PLY subset, not arbitrary Mitsuba-Blender exports. See `README.md:256` and `src/nr_pusch/rt_scene_assets.py:26-35,160-304,320-442,736-793`.

- Scene root: `scene.xml`, exactly `<scene version="2.1.0">`.
- Top-level XML elements: only `bsdf` and `shape`.
- Radio BSDF: `type="itu-radio-material"`; material `type` must be `concrete` or `brick`; thickness is `(0,10]` metres; `scattering_coefficient` must be `0.0`.
- Shape: `type="ply"`; safe ASCII id; exactly one `meshes/<safe-id>.ply` filename, `face_normals=true`, and one `bsdf` reference.
- PLY: ASCII `format ascii 1.0`; only `float x`, `float y`, `float z` vertices and `list uchar int vertex_indices` faces. Faces must be non-degenerate triangles.
- Bake evaluated geometry and object/world transforms into vertex coordinates. PLY coordinates are world XYZ in metres; do not use Mitsuba transforms, `<import>`, custom shape plugins, binary PLY, normals/colors, or external paths.
- Limits: XML 256 KiB; each PLY 8 MiB; expanded scene 32 MiB; 33 files total; 32 shapes and 32 materials; 100,000 vertices and 200,000 faces per PLY; each coordinate component has absolute value at most 10,000 m.
- A scene root may contain only `scene.xml` and the referenced `meshes/*.ply` files. A ZIP must contain those same paths at its root, not an enclosing directory; no README/metadata files inside the validated root or ZIP.

The project importer enforces these constraints in `resolve_scene_assets`, `parse_scene_bundle_zip`, and `_validate_ply`. Keep the simple scene XML in the example or generated package canonical; do not add sensors, integrators, `include`, or plugin-import nodes.

## Conversion workflow

1. **Inspect via MCP.** Open the selected `.blend` without saving. Record scene units, mesh objects, collections, visibility, and world-space bounds. Identify ground separately from buildings and any road/vegetation meshes. Use evaluated mesh data so modifiers are included.
2. **Choose output groups.** Keep ground and buildings separate when practical. Merge building objects into a bounded number of PLYs; never exceed 32 shapes. If a mesh approaches per-file caps, split it into deterministic chunks, still within the total shape/file limits.
3. **Convert coordinates.** Use evaluated mesh vertices transformed by `matrix_world`; apply `scene.unit_settings.scale_length` to get metres. Triangulate with Blender's evaluated loop triangles; preserve winding for negative-determinant transforms and skip only degenerate triangles (cross-product length `<=1e-12`). Verify all coordinates and counts before writing. Do not add Mitsuba-side transforms.
4. **Assign supported radio materials.** Use OSM tags or source material names only when they map defensibly to `concrete` or `brick`. Otherwise choose and state an explicit approximation. This repository cannot represent `metal` or `marble`; never describe mapping them to concrete/brick as physically equivalent.
5. **Write the strict package.** Create a fresh output root with `scene.xml` and `meshes/<safe-id>.ply`; also create a sibling ZIP if the user needs the project's ZIP import UI. Use the exact schema above and check file/ZIP limits.
6. **Keep source unchanged.** Do not save the Blender source or alter OSM files. Report the output paths, exported feature classes, material approximations, units, bounds, mesh counts, and any omitted scene content.

## Verification

1. Validate both folder and ZIP using the repository's helpers:
   - `resolve_scene_assets(output_root, "scene.xml", source="imported")`
   - `parse_scene_bundle_zip(zip_bytes)`
   Confirm each PLY summary is within limits and the ZIP contains only `scene.xml` and referenced mesh files.
2. Load through the project environment with `sionna.rt.load_scene(str(scene_xml), merge_shapes=False)`. `merge_shapes=False` avoids Sionna's default merging by radio material and makes expected shape count/name checks observable; see the [Sionna Scene-Edit tutorial](https://nvlabs.github.io/sionna/rt/tutorials/Scene-Edit.html).
3. Assert the loaded object names/count match the exported shapes; report the Sionna scene type, PLY bounds and triangle/vertex counts. On this workstation, set Mitsuba's `llvm_ad_rgb` variant before importing Sionna RT for CPU-only validation if CUDA/OptiX context errors occur.
4. A native Mitsuba-Blender export is not a substitute: it may contain `<import filename="plugins/__init__.py"/>`, `packed`, or `cycles_lights`, which this project's Sionna importer rejects.

## Validated example from this project

The conversion of `blender_scene/Untitled.blend` produced:

- `blender_scene/test_scene/sionna_rt_export/scene.xml`
- `blender_scene/test_scene/sionna_rt_export/meshes/buildings.ply`
- `blender_scene/test_scene/sionna_rt_export/meshes/ground.ply`
- Sibling ZIP: `blender_scene/test_scene/sionna_rt_export.zip`

The Blender scene contained 153 building meshes and one ground Plane. The
buildings PLY has 3,348 vertices and 4,768 triangles; ground has 4 vertices and
2 triangles. World bounds: X `[-3240.3037, 3240.3037]` m,
Y `[-2906.3228, 2906.3228]` m, Z `[0,78.0319]` m.

Sionna RT 2.2.0 loaded `buildings` and `ground` with `merge_shapes=False`; the
project bundle validator accepted the ZIP. The source scene's visual materials
were `itu_concrete`, `itu_marble`, and `itu_metal`. Because the importer only
accepts concrete/brick radio materials, this conversion mapped building and
ground surfaces to concrete; this is a lossy radio-material approximation.
The Blender scene had no road meshes, so roads were not exported. Sionna emitted
the expected ASCII-PLY slow-parse warning; `buildings.ply` was only 174,231 bytes.

For background on the Blender/OSM workflow, see NVIDIA's [Sionna RT scene creation with Blender and OpenStreetMap](https://www.youtube.com/watch?v=7xHLDxUaQ7c).