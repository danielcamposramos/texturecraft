# Stereo model companion

The native Unix edition offers Enable 3D, Render resolution per eye and Desktop Display Settings in the Viewport menu.
The separate Stereo View is a read-only real-time model companion; the ordinary model viewport retains navigation, editing and picking.
It does not claim to show Rendering mode's final path-traced result.

The default render grid follows the ordinary model viewport's rounded physical pixel edges, preserving its current sampling.
Before that viewport maps, its existing scripted viewport size at the current UI scale supplies the grid.
An explicit width and height remain fixed when the stereo window resizes.
The desktop performs the display scaling once.
The existing renderer maximum of 8192 samples on either axis also bounds the stereo choice.

`viewport.stereo` is inspectable session view state.
Its enabled flag and optional resolution survive a project switch without editing the document or its undo history.
A null resolution restores the viewport-derived grid.

The renderer shares one uploaded world, geometry, materials, textures, lights and light-space shadow cache.
The pure shared camera core makes two parallel off-axis cameras around the existing orbit camera's target plane.
Live compositor limits in native samples bound the baseline, including displacement, visible bake geometry and the perspective background at infinity.
Each eye owns its own temporal accumulation and screen-space effects.
A missing depth budget produces the central camera in both eyes and a visible notice, never an invented separation.

The UI encodes the pair in egui's finish_prepare phase, after ordinary scene uploads.
One pending scene-sync slot can be consumed by the ordinary view or the native callback; its final unapplied owner triggers the existing full-resync mechanism.
Both eyes and their asynchronous readbacks are encoded together from one frame.
Only a complete pair is presented, as one declared full side-by-side buffer, left eye first, with no per-eye resampling.
The shared window owns the native declaration, feedback and display connection.

The desktop-free `o232-scene-renderer` subject contains the shared-camera renderer API, inspectable engine choices and the sibling-path architecture classifier correction.
Only the edition branch adds the native stereo window and its UI controls.
