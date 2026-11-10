# Architecture

TextureCraft is a 3D texture painter: import a mesh, paint PBR material channels on its surface
in layers with masks, bake mesh maps that drive procedural masks, and export game-ready textures.
This document explains how the pieces fit together.

## Crates and layers

```text
L0  core      document model: Project, TextureSet, Layer (paint / fill / group), Mask, Effect,
              Channel, 32 BlendModes with the CPU reference math (blend1/blend3)
L0  texel     TileImage (sparse copy-on-write 64x64 tiles of u16), DilationMap (UV padding),
              deterministic noise, sRGB transfer
L0  geom      Mesh (+ MikkTSpace tangents), primitives, SAH BVH, UvGBuffer, OrbitCamera, auto-unwrap
L1  compose   the layer-stack compositor (CPU, per tile, rayon), patterns, display conversion
L1  bake      AO, curvature, thickness, world normal, position, ID, normal and height bakers
              (rays batched through a Tracer: CPU oracle or the render crate's GPU baker)
L2  paint     strokes -> stamps, camera visibility, texture-space rasterization, commit
L2  formats   mesh import (glTF/GLB, OBJ, PLY, STL), texture export presets, .tcraft projects,
              image resources
L3  engine    Session (document + caches + history + tools), the command registry
L3  render    wgpu viewport renderer (PBR, IBL, post effects, UV view), headless renders,
              GPU path tracing and GPU mesh-map ray casts (compute)
L4  automation MCP server, JSON-lines RPC, bridge to the running app
L5  ui-egui   the egui shell (swappable)
    apps      texturecraft (desktop), texturecraft-cli, texturecraft-web
```

`cargo xtask layers` enforces that a crate depends only on lower layers and that nothing below
`ui-egui` touches egui, eframe, winit or rfd.

## The UV-space G-buffer: one source of truth

For every texture set (one per mesh material) `geom::UvGBuffer` rasterizes the mesh's UV layout
once: each texel knows which triangle covers its center and where on it (barycentrics). From that
every texel has a world position, a shading normal and a tangent frame. Painting, compositing,
procedural masks, baking and UV padding all read the same buffer, so they agree exactly on which
texel is which surface point. Texels whose centers fall outside UV islands are filled by padding
(`texel::DilationMap`, a multi-source BFS computed once per layout).

## Painting

1. A `paint::Stroke` turns pointer samples into world-space stamps: picking through the BVH,
   lazy mouse, spacing measured along the surface (re-projected and re-picked so stamps stay on
   the mesh, resampled in screen space across silhouettes), jitter and mirror symmetry.
2. `paint::rasterize` gathers the triangles within each stamp's volume (BVH box query) and
   accumulates the brush alpha into a texture-space stroke mask for every covered texel inside the
   stamp's oriented box. Because stamps live in 3D, strokes cross UV seams without gaps. A CPU
   eye-distance buffer (`EyeDepth`, rendered once per stroke) keeps paint off surfaces the camera
   can't see; angle fade and backface culling soften grazing surfaces.
3. While dragging, the compositor reads the stroke mask as a live preview. On release,
   `paint::commit` blends it into the layer's tiles with premultiplied "over" (or scales them down
   when erasing); only touched tiles are un-shared. Strokes in the 2D view use UV-space stamps.

## Compositing

`compose::composite_tiles` evaluates a texture set's stack per 64x64 tile in parallel. For each
covered texel it walks the stack bottom to top: paint layers are unpremultiplied from their
(value, coverage) tiles, fill layers evaluate uniform values, procedurals (through UV, tri-planar,
planar, spherical or cylindrical projections) or image resources; masks evaluate their effect
stack (painted bitmaps, fills, mesh-map generators, levels, filters, ID color selection,
compare); a layer's geometry mask (mesh part names) zeroes its mask on texels whose triangle, from
the UV G-buffer, lies in another part; each enabled channel blends with its mode and opacity. Only
dirty tiles are recomposited
(a stroke dirties the tiles it touches), then padded, converted to display formats and uploaded to
the GPU.

The CPU compositor is the reference for everything users see and export. Export
(`formats::export`) reads the same composite, adds height-to-normal (Sobel) combined with the
normal channel and baked normals by reoriented normal mapping, and packs channels per preset.

### Layer-panel thumbnails

The Layers panel shows each layer's content (in the displayed channel), its mask and every effect
stage as small pictures. `compose::thumb::render` runs the same compositor on a *cut* copy of the
project: the requested layer is forced visible at Normal / 100 %, without its mask for content,
and with its effect list truncated after the previewed effect (a hidden effect shows the result
it bypasses). Lower layers remain available when the preview reads anchor points; only the requested
layer is drawn. Instances use their source's channels. Paint and fill layers are flattened to
(value, coverage), so empty texels are transparent; folders composite their children. Colors are
encoded linear to sRGB, data channels raw. Everything is evaluated on a 64 px G-buffer of the mesh
(painted tiles are resampled), never at the set's resolution, and the main composite is never touched.

`engine::thumbs::ThumbService` runs it off the UI thread: an immutable `ThumbSnapshot` (the
project, copy-on-write pixel tiles, baked maps, images and the cached 64 px G-buffer) is rebuilt
only when the document really differs (compared at most every 120 ms while it keeps changing), a
worker thread renders one job at a time (inline, one per frame, on wasm32), results of an older
snapshot are dropped. At most two jobs, including stale generations, retain snapshots in the worker.
Collecting a result requests another frame to display it. An LRU of 96 pictures keeps the out-of-date
picture on screen until its replacement arrives. Only rows on screen ask for pictures; queued rows
that scrolled away are dropped. Baked-map changes are checked by shared buffer identity, without
scanning full-resolution float arrays on the UI thread. Run `perf::bench_layer_thumbs_4k` in release mode to measure snapshot
construction and content, mask and stage rendering on a ten-layer 4096 set. Run
`thumbs::tests::benchmark_unchanged_4k_baked_snapshot` to measure comparisons with 256 MiB of baked maps.

## Undo

`Session::edit` snapshots the `Document` (project + pixels + baked maps + resources) before a
change and restores it if the change fails. Pixel data is copy-on-write (`Arc` tiles), so a
snapshot costs one reference-count bump per allocated tile, and an undo step only keeps the tiles
that actually changed.

## Rendering

`render::Renderer` draws the mesh with metal/roughness PBR: GGX specular with height-correlated
Smith visibility, Lambert diffuse, image-based lighting from procedural studio environments
(prefiltered on the CPU into a mip chain for specular, nine spherical-harmonic coefficients for
diffuse), the analytic split-sum environment BRDF, horizon fading, baked AO, height-to-normal and
the painted normal channel. Debug views show any channel, any baked mesh map or a UV checker.
The 2D view draws the UV layout with the material and a wireframe. Everything is fragment passes,
so it runs on Vulkan, Metal, DX12, WebGPU and WebGL2. `render::headless` renders without a window
(CLI, MCP, tests).

Displacement draws a dense, flat subdivision of the mesh (wgpu has no tessellation stage) and moves
it along the normals by the height channel in the vertex shader of the 3D view and of the
key-light shadow pass (`shaders/displace.wgsl`, shared by both). Rendering mode's path tracers
(`render::trace` on the CPU, `render::gpu_trace` on the GPU) trace the same displaced copy, built
on the CPU by `render::displace` from the composited height.

## MaterialCraft materials

A fill can carry a `MaterialLink` (`core::layer`): a MaterialCraft graph whose cooked maps are the
fill's image sources. `engine::link_cmds` imports `.mcmat` files (`formats::mcmat`), keeps the
`.mcraft` package as a resource, and re-cooks by running `materialcraft-cli` with the link's
values, the texture set's resolution and its baked mesh maps (`LinkJob::run`, on a worker thread
in the UI). Every cook stores its maps under fresh resource names, so undo, redo and the decoded
image cache never disagree. The apps share the file format and the command line, never code.

## Commands and automation

Every action is a `CommandSpec` in `engine::commands` (id, label, menu path, shortcut, params doc,
enabled predicate, run). The menus, shortcuts, the desktop app's control channel
(`--control <port>`), the CLI and the MCP server all execute commands by id, so an agent can do
anything a user can and verify it through `project.inspect`. See
[`control-protocol.md`](control-protocol.md).
