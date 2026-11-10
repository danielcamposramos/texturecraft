# TextureCraft features

Everything TextureCraft does today, by area. What's still missing is in [`ROADMAP.md`](../ROADMAP.md).

## Paint in 3D

- **Brushes** with size, flow, stroke opacity, spacing, hardness, angle and follow path; size,
  angle, flow and position jitter; tangent or camera alignment, angle fade, backface culling, lazy
  mouse, X/Y/Z mirror and radial symmetry (copies paint the far side too) and pen pressure in the
  3D and 2D views (Windows Ink pens, and pens that send touch events in the browser, such as on
  tablets; on Linux, macOS and desktop browsers a pen works as a mouse for now).
  Ctrl+drag in a view sets them on the fly: right-drag the size and hardness, left-drag the flow
  and angle.
  **67 procedural brush alphas** in their own shelf tab, or any imported image as the alpha.
  **18 brush presets** (basic, wear and grunge, details) in the shelf's Brushes tab set the
  stroke and keep the size; save your own brushes there too.
- **Tools:** paint, eraser, **projection** with image stencils (hold S to move, turn and scale the
  stencil), polygon fill (triangle, UV island or the whole mesh), **path** and **lasso fill**, a
  **particle brush**, **smudge**, **clone** (V + click sets the source), color picker, Shift+click
  straight lines, and painting directly in the **2D UV view**.
- **Geometry decals:** drop an image on the surface, then move, scale and turn its frame; it stays
  an editable fill layer.
- **Text** in any installed font (or the built-in ones), shaped for every script with kerning and
  ligatures, one or more lines: stamp it with the brush as its alpha, or place it as a decal in the
  tool's base color (Properties › Alpha › Text).
- **Quick mask** (Y): paint a temporary mask (U to edit, Ctrl+Shift+I to invert) that protects
  everything outside it while you paint.
- **Every channel at once:** base color, metallic, roughness, height, normal, emissive, opacity,
  ambient occlusion and specular level, plus clear coat, sheen, anisotropy, subsurface scattering
  and transmission, each toggled per stroke in the brush's material properties.

## Layers, masks and smart materials

- **Paint, fill, folder and instance layers** with **32 blend modes per channel** (Normal through
  Normal map detail), per-channel opacity, visibility, color tags, drag-and-drop reordering,
  duplicate, group, flatten, cut, copy and paste across texture sets, and **geometry masks** that
  show a layer only on the mesh parts you pick (right-click a layer, or click parts with the
  **Geometry mask** tool, 8).
- **Filters and content effects** on any layer or mask: **31 filters** (HSL, levels,
  contrast/luminosity, color balance, gradient map, posterize, clamp, histogram scan, range and
  shift, invert, threshold, grayscale, replace color, edge detect, normal invert, baked lighting,
  blur, directional and slope blur, sharpen, high pass, bevel, glow, warp, directional warp,
  mosaic and four material finishes), fills and paint. **Levels** shows a histogram of what it
  reads (the mask below it, or a content channel you pick) with its input range marked.
  **Anchor points** record the stack at a layer (or, inside a mask, the mask value at that point)
  so masks, fills and generators higher up can react to it; a generator can take an anchor's
  painted height as **micro detail**. Double-click an effect to rename it.
- **Masks** with a white or black background and a stack of effects: painted strokes, fills,
  **mesh-map generators** (edge wear, dirt, thin areas, curvature, AO, position and direction
  gradients, all with world-space noise), filters, ID color selection and compare. Alt+click a
  mask to view it alone. Drag an effect to reorder its stack, or right-click it to duplicate it
  (painted strokes included) or copy it onto another layer or mask (Ctrl+C, Ctrl+V); with an
  effect selected, Ctrl+D and Delete duplicate or remove it.
- **Fill layers** take uniform values, **23 procedurals** (clouds, cells, grunge, scratches, wood
  grain, bricks, fabric weave, twill, leather grain, marble veins, Voronoi, paint chips…) or your own
  **images**, through UV, tri-planar, planar, spherical, cylindrical or decal projection, with
  the UV scale, rotation and offset, the tri-planar blend sharpness and offset, the planar axis
  and offset, and the cylindrical axis in the fill's properties. Each channel takes its own image (a normal, roughness or height map, say): pick it in the channel's
  source menu, which also imports one, or drop it from the Assets panel onto the channel. A
  normal map image made for DirectX (Y down) is marked with **DirectX normal map** under it.
  Alt+click a material channel (of a fill or the brush) to keep that channel alone.
- **72 materials** (metals from measured reflectance, car paint, brushed anisotropic steel,
  velvet, skin, wax, glass, ice, woods, bamboo, bark, stone, cobblestone, gravel, plaster, sand,
  mud, snow, bone, moss, plastics, fabric, linen, wool, leather, ceramic, rust, cork, paper), **21 smart
  materials** (rusted iron, worn paint, oxidized copper, bronze statue, weathered wood, mossy
  brick and stone, sun-faded paint…) and **24 smart masks**, all procedural, so they work at any
  resolution. Drag them from the tabbed, searchable shelf onto the layer stack or straight onto
  the model.
- **Your own library:** right-click a layer or folder, choose *Create smart material*, and it
  appears under *My smart materials*, pixels and images included.
- **MaterialCraft materials:** *File › Import material*, a drop on the window, or MaterialCraft's
  *File › Send to TextureCraft* (it lands under *My smart materials*) adds a `.mcmat` portable
  material, or a `.mcraft` package, as a fill layer. Its exposed graph inputs (numbers, vectors,
  toggles, options, colors) and presets show in the layer's properties; every change re-cooks
  the maps through MaterialCraft's command line at the texture set's resolution, on a worker
  thread, with the set's baked mesh maps (curvature, AO, thickness…) feeding graph inputs of the
  same name. *Open in MaterialCraft* edits the graph itself. In the browser the cooked maps paint,
  but the inputs are read-only.

## Files and texture sets

- **Open** glTF/GLB, FBX (binary and ASCII), COLLADA, **USD** (usda, usdc, usdz), **Alembic**,
  OBJ, PLY and STL, or a built-in mesh; UVs are generated when a mesh has none. A `.gltf`'s buffers
  and images and a loose USD layer's textures are read from its folder and kept in the project.
  Reimport a changed mesh and your painted strokes stay on the surface.
- **UDIM / UV tiles:** detected on import or switched on later, with a resolution per tile (blurs,
  other filters and height-to-normal slopes keep their size in UV space at any tile resolution)
  and one texture per tile on export (`$udim`).
- **Texture sets** from 128 to 8192 px per material, resampled when you change them, with 21
  channel types to add or remove. Show, hide or invert them all at once, or **focus** on the
  active one. The project's **normal map format** (OpenGL or DirectX, in Texture Set Settings)
  says how a normal map image set as a mesh map is read, and is the default for export.
- **Projects** are a single `.tcraft` file: the mesh, every layer (sparse and lossless), baked maps
  and imported images. **Project templates** start new work from a saved setup. Closing the app,
  or opening, starting or closing another project, with unsaved changes asks whether to save them
  first (in the browser too, where Save downloads the project and closing or reloading the tab
  asks before the changes are lost).

## Bake and export

- **Bake mesh maps** on the GPU (ray traced, about 12× the CPU path) or on every CPU core: ambient
  occlusion, curvature, thickness, world-space normal, position, ID, bent normals, opacity, color
  and direction maps, and normal and height maps from high-poly meshes, with a cage, ray distances
  and matching by mesh name. Or **use your own images** as mesh maps (baked in another app, or
  edited): drop one from the Assets panel on a slot of the Texture Set Settings, or pick it from
  the slot's menu. A UV-tiled set takes one image per tile, named with its tile number
  (`ao.1001.png`, `ao.1002.png`…): drop or pick any one of them.
- **Live bake preview:** while the app bakes, each mesh map shows on the model as soon as it is
  finished; the bake bar follows the newest map, pins a finished one or shows the material.
- **Export presets** for glTF PBR Metal Roughness, Unreal Engine (packed), Unity HDRP (mask map),
  Unity Standard, Blender (Principled BSDF), Godot 4 (ORM), OpenPBR (with the coat, fuzz,
  anisotropy, transmission and subsurface maps), separate maps, mesh maps and document channels, in
  PNG, JPEG, TGA, TIFF, BMP, WEBP, EXR or HDR at 8, 16 or 32 bits, with UV padding, size
  override and OpenGL or DirectX normals.
- **Output template editor:** build your own presets with per-channel packing and converted maps,
  and share them as JSON.
- **Mesh export** to OBJ with an MTL material library and its maps (PBR extension keys for
  roughness, metallic, normal and coat; UV-tiled sets keep their UDIM UVs and export their maps
  through Export textures), textured GLB (with clear coat and sheen) and USD (with clear coat).

## The viewport

- Physically based shading (metallic/roughness or specular/glossiness) with clear coat, sheen,
  anisotropy, subsurface scattering, transmission, parallax occlusion and **displacement** (with
  its own shadows) from the height channel; image-based lighting from six procedural studio and
  outdoor environments or **your own HDR/EXR panoramas**; ACES or linear tone mapping, key-light
  shadows, bloom, vignette, sharpen, grain, **depth of field** and **saved cameras** (also imported
  from glTF, FBX, COLLADA and USD files).
- **Rendering mode** (F10): a progressive, **path-traced** view of your textures with multi-bounce
  light, environment importance sampling, displaced geometry, a denoiser and one-click PNG save,
  traced on the GPU
  (compute shaders, about 25× the CPU tracer, desktop and WebGPU) or on the CPU.
- View any channel, any baked mesh map, or a UV checker; Alt+drag to orbit, middle-drag to pan,
  wheel to zoom, F to frame, M / C / B to cycle views.
- **3D and UV wireframes** that show the mesh's own polygons: quads and n-gons from OBJ, FBX,
  COLLADA, USD, Alembic and PLY files keep their edges, without the triangulation's diagonals.

## Interface

- A modern look in **four brightness levels** (dark, medium dark, medium light and light) with
  **seven accent colors**, set in Settings › Interface.
- **Its own title bar** with the app icon, menus and window buttons on Windows (and on Linux
  desktops that leave the frame to the app, such as GNOME), the macOS traffic lights over the same
  strip, and Settings › Interface › System title bar for the system's frame instead.
- **Dockable, tabbed and floating panels**, saved layouts, and panels that reopen where you closed
  them.
- A **start window** at launch: a new project, a project file, a recent project or the sample
  (Settings › Interface › Start window).
- A **data folder** of your choice for autosaves, smart materials, presets, templates and UI
  layouts, on any drive (Settings › Folders; `Documents/TextureCraft` by default).
- A **shortcut editor** with conflict detection, and every shortcut shown in its tooltip.
- A **History** panel: click any step to go back to it, or forward again to an undone one.
- **10 languages:** English, French, German, Spanish, Portuguese, Italian, Japanese, Chinese,
  Korean and Arabic, switchable live.
- **In the browser:** the same app compiled to WebAssembly, on WebGPU or WebGL2, with touch
  navigation and project open, save and download.

## Built for agents

Everything you can do in the window, an agent can do too:

- **CLI:** `texturecraft-cli new builtin:material_ball out.tcraft`, `bake`, `export`, `render`,
  `run <command> <params>`, `batch`, `commands`, `info`.
- **MCP server:** `texturecraft-cli mcp` gives Claude and other agents tools to create projects,
  add layers and masks, paint strokes, add high polys and a cage, bake, export textures and the
  mesh, reimport a changed mesh, render previews and inspect everything they changed, with file
  access limited to the folders you give it.
- **Control channel:** launch the app with `--control <port>` to drive the running UI: run any
  command, invoke menu items, paint, and take screenshots.

See [`control-protocol.md`](control-protocol.md) for the full protocol and examples.
