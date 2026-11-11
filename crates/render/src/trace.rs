//! Progressive CPU path tracing for Rendering mode: the textured mesh under the environment,
//! with the raster viewport's material conventions (metallic/roughness or specular/glossiness
//! GGX, specular level,
//! height-to-normal slope rule, normal channel blended with RNM, emissive, opacity) but real
//! multi-bounce light transport. Each [`Tracer::pass`] adds one sample per pixel.
//!
//! The optional lobe channels: anisotropic GGX (true anisotropic evaluation and visible-normal
//! sampling along the turned mesh tangent), a GGX clear coat (IOR 1.5) on the mesh normal that
//! takes `1 − coat Fresnel` from everything below it, a Charlie sheen with Estevez–Kulla
//! visibility and albedo scaling of the base, and subsurface scattering as a random walk: the
//! scattering part of the diffuse light enters the mesh, scatters isotropically with the
//! project's per-channel mean free path (no absorption: the base color tints it once, on entry)
//! and leaves where the walk next crosses the surface, as a Lambertian emitter. Walks are capped
//! at [`WALK_STEPS`] (their energy is dropped) and need closed meshes. The clear coat follows the
//! Coat normal channel where a set has it. Transmission turns that part of the dielectric base
//! into a rough glass interface: GGX visible-normal sampling, exact Fresnel at the project's IOR,
//! refraction through the mesh (tinted by the base color on the way in). Depth of field comes from
//! a thin-lens camera ([`TraceScene::lens`]).

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use rayon::prelude::*;
use texturecraft_compose::{Composite, ImageRes};
use texturecraft_core::Channel;
use texturecraft_core::project::{DisplaySettings, ToneMapping};
use texturecraft_geom::glam::{Vec2, Vec3};
use texturecraft_geom::{Bvh, Mesh, OrbitCamera};

use crate::ibl::EnvImage;

/// One texture set's composited channels.
#[derive(Clone)]
pub struct TraceSet {
    /// Mesh material the set covers.
    pub material: usize,
    pub composite: Composite,
    pub height_strength: f32,
}

/// Everything a pass reads (cheap to clone: shared or copy-on-write data).
#[derive(Clone)]
pub struct TraceScene {
    pub mesh: Arc<Mesh>,
    pub bvh: Arc<Bvh>,
    pub sets: Vec<TraceSet>,
    pub env: Arc<EnvImage>,
    /// Environment rotation around +Y (radians), as in the viewport.
    pub env_rotation: f32,
    /// Texture set per triangle (`usize::MAX`: none, default material).
    pub(crate) tri_set: Arc<Vec<usize>>,
    /// Importance sampling of the environment (None for an empty image).
    pub(crate) light: Option<Arc<EnvSampler>>,
    /// Specular/glossiness shading (Specular color is F0, base color is diffuse).
    pub spec_gloss: bool,
    /// Ray offset off surfaces, scaled to the mesh.
    pub(crate) eps: f32,
    /// Subsurface scattering extinction per color channel (1 / mean free path, world units).
    pub(crate) scatter_sigma: Vec3,
    /// Bounding radius of the mesh (scales the scatter radius and the lens).
    pub(crate) radius: f32,
    /// Index of refraction of the Transmission channel.
    pub ior: f32,
    /// Thin lens: focus depth along the view axis (0: the camera's orbit target) and lens radius
    /// as a fraction of the mesh's bounding radius (the viewport's depth of field).
    pub lens: Option<[f32; 2]>,
}

/// Width of the built-in environments traced (texels).
const ENV_SIZE: u32 = 1024;

impl TraceScene {
    pub fn new(mesh: Arc<Mesh>, bvh: Arc<Bvh>, sets: Vec<TraceSet>, env: Arc<EnvImage>, env_rotation: f32) -> TraceScene {
        let mut tri_set = vec![usize::MAX; mesh.triangle_count()];
        let set_of = |material: usize| sets.iter().position(|s| s.material == material).unwrap_or(usize::MAX);
        if mesh.submeshes.is_empty() {
            tri_set.fill(set_of(0));
        }
        for sm in &mesh.submeshes {
            let set = set_of(sm.material);
            let first = sm.start as usize / 3;
            for t in tri_set.iter_mut().skip(first).take(sm.count as usize / 3) {
                *t = set;
            }
        }
        let (lo, hi) = mesh.bounds();
        let eps = ((hi - lo).length() * 2e-5).max(1e-6);
        let light = EnvSampler::new(&env).map(Arc::new);
        let radius = ((hi - lo).length() * 0.5).max(1e-6);
        let mut scene = TraceScene {
            mesh,
            bvh,
            sets,
            env,
            env_rotation,
            tri_set: Arc::new(tri_set),
            light,
            spec_gloss: false,
            eps,
            scatter_sigma: Vec3::ONE,
            radius,
            ior: 1.5,
            lens: None,
        };
        let d = DisplaySettings::default();
        scene.set_scatter(d.scatter_color, d.scatter_radius);
        scene
    }

    /// Subsurface scattering: the mean free path of each color channel is `radius` (a fraction
    /// of the mesh's bounding radius) times that channel of `color`.
    pub fn set_scatter(&mut self, color: [f32; 3], radius: f32) {
        let floor = self.eps * 4.0;
        self.scatter_sigma = Vec3::from(color.map(|c| 1.0 / (radius * c * self.radius).max(floor)));
    }

    /// The scene as `display` shows it: the imported environment `env_image` (when the display
    /// names a resource and it decodes) or the built-in one, its rotation and the shading mode.
    pub fn for_display(mesh: Arc<Mesh>, bvh: Arc<Bvh>, sets: Vec<TraceSet>, display: &DisplaySettings, env_image: Option<&ImageRes>) -> TraceScene {
        TraceScene::for_display_env(mesh, bvh, sets, display, Arc::new(TraceScene::display_env(display, env_image)))
    }

    /// The environment [`TraceScene::for_display`] traces under (building it takes a while: a
    /// caller that rebuilds scenes often keeps it and passes it to [`TraceScene::for_display_env`]).
    pub fn display_env(display: &DisplaySettings, env_image: Option<&ImageRes>) -> EnvImage {
        let imported = env_image.and_then(|i| EnvImage::from_rgba(i.width, i.height, &i.rgba, i.srgb));
        imported.unwrap_or_else(|| crate::ibl::environment(&display.environment, ENV_SIZE))
    }

    /// [`TraceScene::for_display`] under `env`, from [`TraceScene::display_env`] of the same display.
    pub fn for_display_env(mesh: Arc<Mesh>, bvh: Arc<Bvh>, sets: Vec<TraceSet>, display: &DisplaySettings, env: Arc<EnvImage>) -> TraceScene {
        let mut scene = TraceScene::new(mesh, bvh, sets, env, display.env_rotation.to_radians());
        scene.spec_gloss = display.spec_gloss;
        scene.set_scatter(display.scatter_color, display.scatter_radius);
        scene.ior = display.ior;
        scene.lens = display.dof.enabled.then_some([display.dof.focus_distance, display.dof.aperture]);

        scene
    }

    /// The camera ray through `ndc`, through a point of the lens when the scene has one.
    fn camera_ray(&self, cam: &OrbitCamera, ndc: Vec2, aspect: f32, rng: &mut Rng) -> texturecraft_geom::Ray {
        let ray = cam.ray(ndc, aspect);
        let Some([focus, aperture]) = self.lens else { return ray };
        let lens = aperture.max(0.0) * self.radius;
        let fwd = cam.forward();
        let along = ray.dir.dot(fwd);
        if lens <= 0.0 || along <= 1e-6 {
            return ray;
        }
        // The point this pixel sees on the focus plane, viewed from a random point of the lens.
        let depth = if focus > 0.0 { focus } else { cam.distance };
        let focal = ray.origin + ray.dir * ((depth - (ray.origin - cam.eye()).dot(fwd)) / along);
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        let (r, phi) = (lens * rng.next().sqrt(), TAU * rng.next());
        let origin = ray.origin + (right * phi.cos() + up * phi.sin()) * r;
        texturecraft_geom::Ray { origin, dir: (focal - origin).normalize_or(ray.dir) }
    }

    /// World direction to environment space.
    fn to_env(&self, d: Vec3) -> Vec3 {
        let (s, c) = self.env_rotation.sin_cos();
        Vec3::new(c * d.x - s * d.z, d.y, s * d.x + c * d.z)
    }

    fn env(&self, d: Vec3) -> Vec3 {
        self.env.sample(self.to_env(d))
    }

    /// Solid-angle density of [`Self::sample_env`] picking world direction `d`.
    fn env_pdf(&self, d: Vec3) -> f32 {
        self.light.as_ref().map_or(0.0, |l| l.pdf(self.to_env(d)))
    }

    /// A world direction toward the environment, chosen by brightness, with its density.
    fn sample_env(&self, rng: &mut Rng) -> Option<(Vec3, f32)> {
        let (e, pdf) = self.light.as_ref()?.sample(rng.next(), rng.next(), rng.next(), rng.next())?;
        let (s, c) = self.env_rotation.sin_cos();
        Some((Vec3::new(c * e.x + s * e.z, e.y, -s * e.x + c * e.z), pdf))
    }
}

/// Piecewise-constant importance sampling of an equirect environment by luminance and sin(theta).
pub(crate) struct EnvSampler {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Cumulative row probabilities (last = 1).
    pub(crate) rows: Vec<f32>,
    /// Cumulative probabilities within each row (last of each row = 1).
    pub(crate) cols: Vec<f32>,
    /// Probability of each pixel.
    pub(crate) prob: Vec<f32>,
}

impl EnvSampler {
    fn new(env: &EnvImage) -> Option<EnvSampler> {
        let (w, h) = (env.width as usize, env.height as usize);
        if w == 0 || h == 0 || env.data.len() < w * h {
            return None;
        }
        let mut weight: Vec<f32> = env.data.iter().take(w * h).map(|c| luminance(Vec3::from(*c)).max(0.0)).collect();
        for (y, row) in weight.chunks_mut(w).enumerate() {
            let sin = (PI * (y as f32 + 0.5) / h as f32).sin();
            row.iter_mut().for_each(|v| *v = (*v + 1e-4) * sin);
        }
        let total: f32 = weight.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let prob: Vec<f32> = weight.iter().map(|v| v / total).collect();
        let mut rows = Vec::with_capacity(h);
        let mut cols = Vec::with_capacity(w * h);
        let mut acc_rows = 0.0;
        for row in prob.chunks(w) {
            let sum: f32 = row.iter().sum();
            acc_rows += sum;
            rows.push(acc_rows);
            let mut acc = 0.0;
            for p in row {
                acc += if sum > 0.0 { p / sum } else { 1.0 / w as f32 };
                cols.push(acc);
            }
        }
        Some(EnvSampler { width: w, height: h, rows, cols, prob })
    }

    /// Environment-space direction and its solid-angle density.
    fn sample(&self, u0: f32, u1: f32, u2: f32, u3: f32) -> Option<(Vec3, f32)> {
        let y = self.rows.partition_point(|c| *c < u0 * self.rows.last().copied().unwrap_or(1.0)).min(self.height - 1);
        let row = self.cols.get(y * self.width..(y + 1) * self.width)?;
        let x = row.partition_point(|c| *c < u1 * row.last().copied().unwrap_or(1.0)).min(self.width - 1);
        let d = crate::ibl::dir_of((x as f32 + u2) / self.width as f32, (y as f32 + u3) / self.height as f32);
        let pdf = self.pdf(d);
        (pdf > 0.0).then_some((d, pdf))
    }

    fn pdf(&self, d: Vec3) -> f32 {
        let uv = crate::ibl::uv_of(d);
        let x = ((uv.x * self.width as f32) as usize).min(self.width - 1);
        let y = ((uv.y * self.height as f32) as usize).min(self.height - 1);
        let sin = (PI * (y as f32 + 0.5) / self.height as f32).sin();
        let p = self.prob.get(y * self.width + x).copied().unwrap_or(0.0);
        if sin <= 1e-6 { 0.0 } else { p * (self.width * self.height) as f32 / (2.0 * PI * PI * sin) }
    }
}

/// How the accumulated radiance becomes display pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceDisplay {
    /// Exposure multiplier (2^EV).
    pub exposure: f32,
    pub aces: bool,
    /// How much of the environment shows behind the mesh (0: studio backdrop).
    pub env_opacity: f32,
    /// Filter the remaining noise (edge-aware; keeps texture detail).
    pub denoise: bool,
}

impl TraceDisplay {
    /// Exposure, tone mapping and background as the viewport shows them.
    pub fn from_display(d: &DisplaySettings) -> TraceDisplay {
        TraceDisplay { exposure: d.env_exposure.exp2(), aces: d.tone_mapping == ToneMapping::AcesFilmic, env_opacity: d.env_opacity, denoise: false }
    }
}

/// Light transport settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceSettings {
    /// Diffuse/specular bounces after the camera hit.
    pub bounces: u32,
}

impl Default for TraceSettings {
    fn default() -> Self {
        TraceSettings { bounces: 6 }
    }
}

#[derive(Clone, Copy, Default)]
struct Px {
    /// Radiance summed over samples that hit the mesh.
    hit: Vec3,
    /// Sum of the squared luminance of those samples (for the variance).
    hit_lum2: f32,
    /// First-hit reflectance, shading normal and distance, summed (denoiser guides).
    albedo: Vec3,
    normal: Vec3,
    depth: f32,
    /// Environment radiance summed over samples that missed it.
    env: Vec3,
    hits: f32,
}

/// What one camera sample saw.
struct Sample {
    radiance: Vec3,
    albedo: Vec3,
    normal: Vec3,
    depth: f32,
}

/// The accumulation buffer of one view.
pub struct Tracer {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// First row of the pass in progress (see [`Tracer::pass_rows`]).
    row: u32,
    acc: Vec<Px>,
}

impl Tracer {
    pub fn new(width: u32, height: u32) -> Tracer {
        let (w, h) = (width.clamp(1, 8192), height.clamp(1, 8192));
        Tracer { width: w, height: h, samples: 0, row: 0, acc: vec![Px::default(); (w * h) as usize] }
    }

    /// Add one sample per pixel.
    pub fn pass(&mut self, scene: &TraceScene, cam: &OrbitCamera, settings: &TraceSettings) {
        while !self.pass_rows(scene, cam, settings, self.height) {}
    }

    /// Sample up to `rows` more rows of the current pass; true when that finished the pass (the
    /// new sample then counts in [`Tracer::samples`]). Lets a single-threaded caller spread a
    /// pass over several frames.
    pub fn pass_rows(&mut self, scene: &TraceScene, cam: &OrbitCamera, settings: &TraceSettings, rows: u32) -> bool {
        let (w, h, pass) = (self.width, self.height, self.samples);
        let aspect = w as f32 / h as f32;
        let (y0, y1) = (self.row, self.row.saturating_add(rows.max(1)).min(h));
        let span = self.acc.get_mut(y0 as usize * w as usize..y1 as usize * w as usize).unwrap_or_default();
        span.par_chunks_mut(w as usize).enumerate().for_each(|(dy, row)| {
            let y = y0 as usize + dy;
            for (x, px) in row.iter_mut().enumerate() {
                let mut rng = Rng::new(x as u32, y as u32, pass);
                let ndc = Vec2::new((x as f32 + rng.next()) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + rng.next()) / h as f32 * 2.0);
                let ray = scene.camera_ray(cam, ndc, aspect, &mut rng);
                match trace(scene, ray.origin, ray.dir, settings, &mut rng) {
                    Some(s) => {
                        px.hit += s.radiance;
                        px.hit_lum2 += luminance(s.radiance).powi(2);
                        px.albedo += s.albedo;
                        px.normal += s.normal;
                        px.depth += s.depth;
                        px.hits += 1.0;
                    }
                    None => px.env += scene.env(ray.dir),
                }
            }
        });
        if y1 >= h {
            self.row = 0;
            self.samples += 1;
            true
        } else {
            self.row = y1;
            false
        }
    }

    /// Tone-mapped sRGB pixels, row-major from the top.
    pub fn rgba8(&self, d: &TraceDisplay) -> Vec<u8> {
        let n = self.samples.max(1) as f32;
        let (w, h) = (self.width as f32, self.height as f32);
        let denoised = if d.denoise { self.denoise() } else { Vec::new() };
        let mut out = vec![0u8; self.acc.len() * 4];
        out.par_chunks_mut(4).zip(self.acc.par_iter()).enumerate().for_each(|(i, (o, px))| {
            let uv = Vec2::new(((i as u32 % self.width) as f32 + 0.5) / w, ((i as u32 / self.width) as f32 + 0.5) / h);
            let a = px.hits / n;
            let misses = n - px.hits;
            let back = if misses > 0.0 { backdrop(uv).lerp(tonemap(px.env / misses, d), d.env_opacity.clamp(0.0, 1.0)) } else { backdrop(uv) };
            let mean = denoised.get(i).copied().unwrap_or(px.hit / px.hits.max(1.0));
            let scene = if px.hits > 0.0 { tonemap(mean, d) } else { Vec3::ZERO };
            let c = back.lerp(scene, a);
            for (k, v) in c.to_array().into_iter().enumerate() {
                if let Some(o) = o.get_mut(k) {
                    *o = (texturecraft_texel::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                }
            }
            if let Some(o) = o.get_mut(3) {
                *o = 255;
            }
        });
        out
    }
}

impl Tracer {
    /// Edge-avoiding à-trous filtering of the mesh radiance: lighting is divided by the first-hit
    /// reflectance (so texture detail stays sharp), filtered five times with growing steps, its
    /// weights falling off across normal, depth and luminance edges (scaled by the estimated
    /// noise), then multiplied back. Returns the mean radiance per pixel.
    fn denoise(&self) -> Vec<Vec3> {
        let (w, h) = (self.width as usize, self.height as usize);
        struct G {
            illum: Vec3,
            albedo: Vec3,
            normal: Vec3,
            depth: f32,
            var: f32,
            hit: bool,
        }
        let guides: Vec<G> = self
            .acc
            .par_iter()
            .map(|px| {
                let n = px.hits.max(1.0);
                let albedo = (px.albedo / n).max(Vec3::splat(0.02));
                let mean = px.hit / n;
                let var = (px.hit_lum2 / n - luminance(mean).powi(2)).max(0.0) / n;
                let a = luminance(albedo).max(0.02);
                G { illum: mean / albedo, albedo, normal: px.normal.normalize_or(Vec3::Z), depth: px.depth / n, var: var / (a * a), hit: px.hits > 0.0 }
            })
            .collect();
        const H: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];
        let mut cur: Vec<(Vec3, f32)> = guides.iter().map(|g| (g.illum, g.var)).collect();
        for step in [1usize, 2, 4, 8, 16] {
            let prev = cur;
            cur = (0..w * h)
                .into_par_iter()
                .map(|i| {
                    let (Some(p), Some(&(ip, vp))) = (guides.get(i), prev.get(i)) else { return (Vec3::ZERO, 0.0) };
                    if !p.hit {
                        return (ip, vp);
                    }
                    let (x, y) = ((i % w) as i64, (i / w) as i64);
                    let sigma_l = 4.0 * vp.sqrt() + 1e-3;
                    let (mut sum, mut wsum, mut vsum) = (Vec3::ZERO, 0.0f32, 0.0f32);
                    for (j, hy) in H.iter().enumerate() {
                        for (k, hx) in H.iter().enumerate() {
                            let qx = x + (k as i64 - 2) * step as i64;
                            let qy = y + (j as i64 - 2) * step as i64;
                            if qx < 0 || qy < 0 || qx >= w as i64 || qy >= h as i64 {
                                continue;
                            }
                            let qi = qy as usize * w + qx as usize;
                            let (Some(q), Some(&(iq, vq))) = (guides.get(qi), prev.get(qi)) else { continue };
                            if !q.hit {
                                continue;
                            }
                            let wn = p.normal.dot(q.normal).max(0.0).powi(64);
                            let wz = (-(p.depth - q.depth).abs() / (0.01 * p.depth.max(1e-3) * step as f32)).exp();
                            let wl = (-(luminance(ip) - luminance(iq)).abs() / sigma_l).exp();
                            let wt = hx * hy * wn * wz * wl;
                            sum += iq * wt;
                            wsum += wt;
                            vsum += wt * wt * vq;
                        }
                    }
                    if wsum > 1e-8 { (sum / wsum, vsum / (wsum * wsum)) } else { (ip, vp) }
                })
                .collect();
        }
        cur.iter().zip(&guides).map(|((illum, _), g)| *illum * g.albedo).collect()
    }
}

/// The viewport's studio backdrop: a cool slate, lighter at the top (matches the post pass).
fn backdrop(uv: Vec2) -> Vec3 {
    let d = uv * 2.0 - Vec2::ONE;
    Vec3::new(0.0295, 0.0343, 0.0437).lerp(Vec3::new(0.0110, 0.0130, 0.0168), uv.y) * (1.0 - 0.09 * d.length_squared())
}

fn tonemap(hdr: Vec3, d: &TraceDisplay) -> Vec3 {
    let x = hdr.max(Vec3::ZERO) * d.exposure;
    if !d.aces {
        return x.min(Vec3::ONE);
    }
    let v = x * 0.6;
    ((v * (v * 2.51 + 0.03)) / (v * (v * 2.43 + 0.59) + 0.14)).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Small, fast per-sample generator (PCG hash seeded by pixel and pass).
struct Rng(u32);

impl Rng {
    fn new(x: u32, y: u32, pass: u32) -> Rng {
        Rng(texturecraft_texel::noise::pcg3d(x, y, pass.wrapping_mul(7919).wrapping_add(17))[0] | 1)
    }

    fn next(&mut self) -> f32 {
        // xorshift32
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Shading data at a hit.
struct Surface {
    pos: Vec3,
    /// Geometric normal facing the incoming ray.
    geo: Vec3,
    n: Vec3,
    /// Anisotropy direction: the mesh tangent turned by the anisotropy angle, orthogonal to `n`.
    tangent: Vec3,
    /// Diffuse albedo and specular reflectance at normal incidence.
    diffuse: Vec3,
    f0: Vec3,
    roughness: f32,
    emissive: Vec3,
    opacity: f32,
    anisotropy: f32,
    coat: f32,
    coat_roughness: f32,
    /// The interpolated mesh normal (the clear coat ignores normal and height channels).
    coat_n: Vec3,
    /// Sheen color × weight.
    sheen: Vec3,
    sheen_roughness: f32,
    scatter: f32,
    /// Transmissive part of the base, its tint (the base color), and whether the ray enters the
    /// mesh here (it hit the outside of the triangle).
    transmission: f32,
    base: Vec3,
    front: bool,
}

fn lerp3(a: Vec3, b: Vec3, c: Vec3, u: f32, v: f32) -> Vec3 {
    a * (1.0 - u - v) + b * u + c * v
}

/// Tangent-space normal blending (reoriented normal mapping), as in the viewport shader.
fn rnm(base: Vec3, detail: Vec3) -> Vec3 {
    let t = base + Vec3::Z;
    let u = detail * Vec3::new(-1.0, -1.0, 1.0);
    (t * t.dot(u) - u * t.z).normalize_or(Vec3::Z)
}

fn surface(scene: &TraceScene, tri: u32, u: f32, v: f32, o: Vec3, d: Vec3, t: f32) -> Option<Surface> {
    let m = &scene.mesh;
    let i = tri as usize * 3;
    let idx = [*m.indices.get(i)? as usize, *m.indices.get(i + 1)? as usize, *m.indices.get(i + 2)? as usize];
    let p = idx.map(|k| m.positions.get(k).copied().unwrap_or(Vec3::ZERO));
    let mut geo = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Y);
    let nrm = idx.map(|k| m.normals.get(k).copied().unwrap_or(geo));
    let mut n_geo = lerp3(nrm[0], nrm[1], nrm[2], u, v).normalize_or(geo);
    let front = geo.dot(d) <= 0.0;
    if !front {
        geo = -geo;
    }
    if n_geo.dot(geo) < 0.0 {
        n_geo = -n_geo;
    }
    let uvs = idx.map(|k| m.uvs.get(k).copied().unwrap_or(Vec2::ZERO));
    let uv = uvs[0] * (1.0 - u - v) + uvs[1] * u + uvs[2] * v;
    let tan = idx.map(|k| m.tangents.get(k).copied().unwrap_or(texturecraft_geom::glam::Vec4::X));
    let tw = lerp3(tan[0].truncate(), tan[1].truncate(), tan[2].truncate(), u, v);
    let sign = if tan[0].w < 0.0 { -1.0 } else { 1.0 };

    let set = scene.tri_set.get(tri as usize).and_then(|s| scene.sets.get(*s));
    // The composite texel at offset (dx, dy) from the hit.
    let texel = |s: &TraceSet, dx: i32, dy: i32| {
        let (w, h) = (s.composite.resolution.max(1) as i32, s.composite.height.max(1) as i32);
        (((uv.x * w as f32).floor() as i32 + dx).rem_euclid(w) as u32, ((uv.y * h as f32).floor() as i32 + dy).rem_euclid(h) as u32)
    };
    let ch = |c: Channel, dx: i32, dy: i32| -> [f32; 3] {
        let Some(s) = set else { return c.default_value() };
        let (x, y) = texel(s, dx, dy);
        s.composite.value(c, x, y)
    };
    let base = Vec3::from(ch(Channel::BaseColor, 0, 0));
    // A UV tile smaller than its atlas block has fewer texels per UV unit: slopes per texel shrink.
    let k = set.map_or(0.0, |s| {
        let (x, y) = texel(s, 0, 0);
        s.height_strength * s.composite.resolution as f32 / 16.0 * s.composite.block_scales.at(x, y)
    });
    let n_height = if k != 0.0 {
        let h = |dx, dy| ch(Channel::Height, dx, dy)[0];
        Vec3::new(-(h(1, 0) - h(-1, 0)) * 0.5 * k, (h(0, 1) - h(0, -1)) * 0.5 * k, 1.0).normalize_or(Vec3::Z)
    } else {
        Vec3::Z
    };
    let detail = (Vec3::from(ch(Channel::Normal, 0, 0)) * 2.0 - Vec3::ONE).normalize_or(Vec3::Z);
    let n_ts = rnm(n_height, detail);
    let tt = (tw - n_geo * n_geo.dot(tw)).normalize_or(n_geo.any_orthonormal_vector());
    let b = n_geo.cross(tt) * sign;
    let mut n = (tt * n_ts.x + b * n_ts.y + n_geo * n_ts.z).normalize_or(n_geo);
    if n.dot(-d) <= 0.0 {
        n = n_geo;
    }
    let scalar = |c: Channel| ch(c, 0, 0)[0].clamp(0.0, 1.0);
    let (sin, cos) = (scalar(Channel::AnisotropyAngle) * PI).sin_cos();
    let turned = tt * cos + b * sin;
    let tangent = (turned - n * n.dot(turned)).normalize_or(n.any_orthonormal_vector());
    // Glossiness replaces roughness wherever a set has it (as in the viewport).
    let roughness = match set {
        Some(s) if s.composite.channels.contains_key(&Channel::Glossiness) => 1.0 - ch(Channel::Glossiness, 0, 0)[0],
        _ => ch(Channel::Roughness, 0, 0)[0],
    };
    let (diffuse, f0, metallic) = if scene.spec_gloss {
        let f0 = Vec3::from(ch(Channel::Specular, 0, 0)).clamp(Vec3::ZERO, Vec3::ONE);
        (base * (1.0 - f0.max_element()), f0, 0.0)
    } else {
        let metallic = ch(Channel::Metallic, 0, 0)[0].clamp(0.0, 1.0);
        let level = ch(Channel::SpecularLevel, 0, 0)[0].clamp(0.0, 1.0);
        (base * (1.0 - metallic), Vec3::splat(0.08 * level).lerp(base, metallic), metallic)
    };
    let coat_base = if n_geo.dot(-d) > 0.0 { n_geo } else { geo };
    let coat_n = match set {
        Some(s) if s.composite.channels.contains_key(&Channel::CoatNormal) => {
            let c = (Vec3::from(ch(Channel::CoatNormal, 0, 0)) * 2.0 - Vec3::ONE).normalize_or(Vec3::Z);
            let turned = (tt * c.x + b * c.y + coat_base * c.z).normalize_or(coat_base);
            if turned.dot(-d) > 0.0 { turned } else { coat_base }
        }
        _ => coat_base,
    };
    Some(Surface {
        pos: o + d * t,
        geo,
        n,
        tangent,
        diffuse,
        f0,
        roughness: roughness.clamp(0.045, 1.0),
        emissive: Vec3::from(ch(Channel::Emissive, 0, 0)),
        opacity: ch(Channel::Opacity, 0, 0)[0].clamp(0.0, 1.0),
        anisotropy: scalar(Channel::AnisotropyLevel),
        coat: scalar(Channel::CoatWeight),
        coat_roughness: scalar(Channel::CoatRoughness).max(0.045),
        coat_n,
        sheen: Vec3::from(ch(Channel::SheenColor, 0, 0)).clamp(Vec3::ZERO, Vec3::ONE) * scalar(Channel::SheenWeight),
        sheen_roughness: scalar(Channel::SheenRoughness).max(0.07),
        scatter: scalar(Channel::ScatterWeight),
        transmission: scalar(Channel::Transmission) * (1.0 - metallic),
        base: base.clamp(Vec3::ZERO, Vec3::ONE),
        front,
    })
}

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

fn schlick(f0: Vec3, cos: f32) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos.clamp(0.0, 1.0)).powi(5)
}

/// Fresnel of the clear coat (IOR 1.5).
fn coat_fresnel(cos: f32) -> f32 {
    schlick(Vec3::splat(0.04), cos).x
}

/// Cosine-weighted direction around local +z.
fn cosine_local(u1: f32, u2: f32) -> Vec3 {
    let r = u1.sqrt();
    let phi = TAU * u2;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).max(0.0).sqrt())
}

/// Cosine-weighted world direction around `n`.
fn cosine_around(n: Vec3, rng: &mut Rng) -> Vec3 {
    let (t, b) = n.any_orthonormal_pair();
    let l = cosine_local(rng.next(), rng.next());
    t * l.x + b * l.y + n * l.z
}

/// Smith GGX masking for one direction (local frame, z = normal), anisotropic alphas.
fn smith_g1(w: Vec3, at: f32, ab: f32) -> f32 {
    let z = w.z.max(1e-6);
    2.0 * z / (z + (at * at * w.x * w.x + ab * ab * w.y * w.y + z * z).sqrt())
}

/// GGX visible-normal sampling (Heitz 2018; local frame, z = normal), anisotropic alphas.
fn sample_vndf(ve: Vec3, at: f32, ab: f32, u1: f32, u2: f32) -> Vec3 {
    let vh = Vec3::new(at * ve.x, ab * ve.y, ve.z).normalize_or(Vec3::Z);
    let len2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = if len2 > 0.0 { Vec3::new(-vh.y, vh.x, 0.0) / len2.sqrt() } else { Vec3::X };
    let t2 = vh.cross(t1);
    let r = u1.sqrt();
    let phi = TAU * u2;
    let p1 = r * phi.cos();
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * r * phi.sin();
    let nh = t1 * p1 + t2 * p2 + vh * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    Vec3::new(at * nh.x, ab * nh.y, nh.z.max(1e-6)).normalize_or(Vec3::Z)
}

/// A GGX microfacet lobe in its own frame (z = normal, x = the direction of `at`).
struct Ggx {
    t: Vec3,
    b: Vec3,
    n: Vec3,
    /// View direction, local.
    v: Vec3,
    at: f32,
    ab: f32,
}

impl Ggx {
    fn new(t: Vec3, n: Vec3, v: Vec3, at: f32, ab: f32) -> Ggx {
        let b = n.cross(t);
        Ggx { t, b, n, v: Vec3::new(v.dot(t), v.dot(b), v.dot(n).max(1e-4)), at: at.max(1e-3), ab: ab.max(1e-3) }
    }

    fn local(&self, w: Vec3) -> Vec3 {
        Vec3::new(w.dot(self.t), w.dot(self.b), w.dot(self.n))
    }

    fn d(&self, h: Vec3) -> f32 {
        let k = (h.x / self.at).powi(2) + (h.y / self.ab).powi(2) + h.z * h.z;
        1.0 / (PI * self.at * self.ab * k * k).max(1e-12)
    }

    /// D·G / (4 cos θv) toward world direction `w` (the BRDF × cos θl without Fresnel), the
    /// density of [`Self::sample`] picking `w`, and cos(l, h) for the Fresnel term.
    fn eval(&self, w: Vec3) -> Option<(f32, f32, f32)> {
        let l = self.local(w);
        if l.z <= 0.0 {
            return None;
        }
        let h = (l + self.v).normalize_or(Vec3::Z);
        let d = self.d(h);
        let g1v = smith_g1(self.v, self.at, self.ab);
        let pdf = g1v * d / (4.0 * self.v.z);
        Some((pdf * smith_g1(l, self.at, self.ab), pdf, l.dot(h)))
    }

    /// A world direction reflected off a visible normal.
    fn sample(&self, u1: f32, u2: f32) -> Option<Vec3> {
        let h = sample_vndf(self.v, self.at, self.ab, u1, u2);
        let l = h * 2.0 * self.v.dot(h) - self.v;
        (l.z > 0.0).then(|| self.t * l.x + self.b * l.y + self.n * l.z)
    }
}

/// Sheen distribution ("Charlie"), alpha = sheen roughness².
fn d_charlie(a: f32, noh: f32) -> f32 {
    let inv = 1.0 / a;
    (2.0 + inv) * (1.0 - noh * noh).max(1e-4).powf(inv * 0.5) / TAU
}

/// Estevez & Kulla's fitted Λ of the sheen visibility ("Production Friendly Microfacet Sheen
/// BRDF", 2017).
fn sheen_lambda(cos: f32, a: f32) -> f32 {
    let o = (1.0 - a) * (1.0 - a);
    let mix = |x: f32, y: f32| x + (y - x) * o;
    let l = |x: f32| {
        mix(21.5473, 25.3245) / (1.0 + mix(3.82987, 3.32435) * x.max(1e-4).powf(mix(0.19823, 0.16801))) + mix(-1.97760, -1.27393) * x + mix(-4.32054, -4.85967)
    };
    if cos < 0.5 { l(cos).exp() } else { (2.0 * l(0.5) - l(1.0 - cos)).exp() }
}

fn sheen_vis(nov: f32, nol: f32, a: f32) -> f32 {
    (1.0 / ((1.0 + sheen_lambda(nov, a) + sheen_lambda(nol, a)) * 4.0 * nov * nol.max(1e-4))).clamp(0.0, 1.0)
}

/// Directional albedo of the sheen lobe at sheen roughness `r` (the viewport's fit, see
/// `sheen_albedo` in mesh.wgsl).
fn sheen_albedo(nov: f32, r: f32) -> f32 {
    let a = 1.2416 - 1.7580 * r + 1.7164 * r * r;
    let b = 6.2757 / (r + 0.2077) - 8.6381 + 4.4979 * r;
    let q = 1.2287 - 0.5161 * r;
    let c = 0.1121 * r - 0.3840 * r * r;
    (a * (-b * nov.max(0.0).powf(q)).exp() + c).clamp(0.0, 1.0)
}

/// The surface's BRDF at one hit: anisotropic GGX specular and Lambert diffuse, under a sheen
/// and a clear coat. The scattering part of the diffuse light is not in it: see
/// [`Bsdf::walk_albedo`].
struct Bsdf {
    spec: Ggx,
    coat: Ggx,
    f0: Vec3,
    /// Diffuse albedo that reflects at the surface.
    diffuse: Vec3,
    /// Fresnel at the view angle.
    fv: Vec3,
    coat_weight: f32,
    /// Energy the coat lets through toward the view.
    coat_t: f32,
    sheen: Vec3,
    sheen_a: f32,
    sheen_r: f32,
    /// Sheen albedo toward the view.
    sheen_ev: f32,
    /// Probabilities of sampling the specular, diffuse, sheen and coat lobes.
    p: [f32; 4],
    /// Albedo of the diffuse light that scatters under the surface, and the probability of
    /// following it with a random walk instead of sampling this BRDF.
    walk_albedo: Vec3,
    p_walk: f32,
}

impl Bsdf {
    fn new(s: &Surface, v: Vec3) -> Bsdf {
        let a = (s.roughness * s.roughness).max(1e-3);
        let spec = Ggx::new(s.tangent, s.n, v, a * (1.0 + s.anisotropy), a * (1.0 - s.anisotropy));
        let coat_a = s.coat_roughness * s.coat_roughness;
        let coat = Ggx::new(s.coat_n.any_orthonormal_vector(), s.coat_n, v, coat_a, coat_a);
        let fv = schlick(s.f0, spec.v.z);
        let coat_t = 1.0 - s.coat * coat_fresnel(coat.v.z);
        let sheen_ev = sheen_albedo(spec.v.z, s.sheen_roughness);
        let base = coat_t * (1.0 - s.sheen.max_element() * sheen_ev);
        let diffuse = s.diffuse * (1.0 - s.scatter);
        let walk_albedo = s.diffuse * s.scatter * (Vec3::ONE - fv) * base;
        let w = [luminance(fv) * base, luminance(diffuse * (Vec3::ONE - fv)) * base, luminance(s.sheen) * sheen_ev * coat_t, s.coat * coat_fresnel(coat.v.z)];
        let sum: f32 = w.iter().sum();
        let w = w.map(|x| if x > 1e-5 { x.max(0.1 * sum) } else { 0.0 });
        let total: f32 = w.iter().sum();
        let p = if total > 0.0 { w.map(|x| x / total) } else { [1.0, 0.0, 0.0, 0.0] };
        let walk = luminance(walk_albedo);
        let p_walk = if walk > 1e-5 { (walk / (walk + sum)).clamp(0.05, 0.95) } else { 0.0 };
        Bsdf {
            spec,
            coat,
            f0: s.f0,
            diffuse,
            fv,
            coat_weight: s.coat,
            coat_t,
            sheen: s.sheen,
            sheen_a: s.sheen_roughness * s.sheen_roughness,
            sheen_r: s.sheen_roughness,
            sheen_ev,
            p,
            walk_albedo,
            p_walk,
        }
    }

    /// BRDF times cos(theta) toward world direction `w`, and the density of [`Self::sample`]
    /// picking it.
    fn eval(&self, w: Vec3) -> (Vec3, f32) {
        let (mut f, mut pdf) = (Vec3::ZERO, 0.0);
        let l = self.spec.local(w);
        if l.z > 0.0 {
            // Sheen albedo scaling (the glTF sheen convention): energy the sheen reflects
            // toward the view or the light never reaches the base.
            let m = self.sheen.max_element();
            let base = self.coat_t * if m > 0.0 { (1.0 - m * self.sheen_ev).min(1.0 - m * sheen_albedo(l.z, self.sheen_r)) } else { 1.0 };
            if let Some((g, p, cos_lh)) = self.spec.eval(w) {
                f += schlick(self.f0, cos_lh) * g * base;
                pdf += self.p[0] * p;
            }
            f += self.diffuse * (Vec3::ONE - self.fv) * (l.z / PI * base);
            pdf += (self.p[1] + self.p[2]) * l.z / PI;
            if m > 0.0 {
                let h = (l + self.spec.v).normalize_or(Vec3::Z);
                f += self.sheen * (d_charlie(self.sheen_a, h.z) * sheen_vis(self.spec.v.z, l.z, self.sheen_a) * l.z * self.coat_t);
            }
        }
        if self.coat_weight > 0.0
            && let Some((g, p, cos_lh)) = self.coat.eval(w)
        {
            f += Vec3::splat(self.coat_weight * coat_fresnel(cos_lh) * g);
            pdf += self.p[3] * p;
        }
        (f, pdf)
    }

    /// A world direction from the BRDF lobes (specular lobes by visible normals, diffuse and
    /// sheen by cosine).
    fn sample(&self, rng: &mut Rng) -> Option<Vec3> {
        let (u0, u1, u2) = (rng.next(), rng.next(), rng.next());
        let [ps, pd, pz, _] = self.p;
        if u0 < ps {
            self.spec.sample(u1, u2)
        } else if u0 < ps + pd + pz {
            let l = cosine_local(u1, u2);
            Some(self.spec.t * l.x + self.spec.b * l.y + self.spec.n * l.z)
        } else {
            self.coat.sample(u1, u2)
        }
    }
}

/// Unpolarized Fresnel reflectance of a dielectric interface; `eta` = n(incident) / n(other side).
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (rs * rs + rp * rp)
}

/// Sampling density standing in for a lobe that next-event estimation never samples (its
/// environment hits take the full MIS weight).
const UNSAMPLED_PDF: f32 = 1e15;

/// The transmissive base as a rough dielectric interface (GGX, visible normals): reflect with
/// the Fresnel at `ior`, else refract into (or out of) the mesh. Returns the new direction and the
/// path weight (G1 of the new direction; the base color tints light entering the mesh).
/// shortcut: radiance is not scaled by the squared IOR ratio at each crossing (the factors cancel
/// for paths that leave the mesh again).
fn dielectric(s: &Surface, v: Vec3, ior: f32, rng: &mut Rng) -> Option<(Vec3, Vec3)> {
    let a = (s.roughness * s.roughness).max(1e-3);
    let lobe = Ggx::new(s.tangent, s.n, v, a, a);
    let hl = sample_vndf(lobe.v, a, a, rng.next(), rng.next());
    let h = lobe.t * hl.x + lobe.b * hl.y + lobe.n * hl.z;
    let cos_i = v.dot(h).max(1e-6);
    let eta = if s.front { 1.0 / ior.max(1.0) } else { ior.max(1.0) };
    if rng.next() < fresnel_dielectric(cos_i, eta) {
        let l = h * 2.0 * cos_i - v;
        let ll = lobe.local(l);
        return (l.dot(s.geo) > 0.0 && ll.z > 0.0).then(|| (l, Vec3::splat(smith_g1(ll, a, a))));
    }
    let cos_t = (1.0 - eta * eta * (1.0 - cos_i * cos_i)).max(0.0).sqrt();
    let l = (h * (eta * cos_i - cos_t) - v * eta).normalize_or(-v);
    let ll = lobe.local(l);
    let tint = if s.front { s.base } else { Vec3::ONE };
    (l.dot(s.geo) < 0.0 && ll.z < 0.0).then(|| (l, tint * smith_g1(-ll, a, a)))
}

/// Most scattering events a subsurface random walk takes before giving up.
const WALK_STEPS: usize = 48;

/// Follow light under the surface at `s`: enter by a cosine-weighted direction, then fly
/// exponential distances (one color channel's extinction picked per step, weighted by the
/// average density over the channels) and scatter isotropically until the path leaves the
/// mesh. Returns the exit point, the outward geometric normal there and the per-channel weight.
fn random_walk(scene: &TraceScene, s: &Surface, rng: &mut Rng) -> Option<(Vec3, Vec3, Vec3)> {
    let sigma = scene.scatter_sigma;
    let mean = |v: Vec3| (v.x + v.y + v.z) / 3.0;
    let mut p = s.pos - s.geo * scene.eps;
    let mut d = cosine_around(-s.geo, rng);
    let mut w = Vec3::ONE;
    for _ in 0..WALK_STEPS {
        let c = ((rng.next() * 3.0) as usize).min(2);
        let t = -(1.0 - rng.next()).max(1e-12).ln() / sigma[c];
        if let Some(hit) = scene.bvh.closest(p, d, t) {
            let tr = (-sigma * hit.t).exp();
            w *= tr / mean(tr).max(1e-30);
            let m = &scene.mesh;
            let i = hit.tri as usize * 3;
            let q = [m.indices.get(i)?, m.indices.get(i + 1)?, m.indices.get(i + 2)?].map(|k| m.positions.get(*k as usize).copied().unwrap_or(Vec3::ZERO));
            let n = (q[1] - q[0]).cross(q[2] - q[0]).normalize_or(-d);
            let out = if n.dot(d) > 0.0 { n } else { -n };
            return w.is_finite().then_some((p + d * hit.t, out, w));
        }
        let tr = (-sigma * t).exp();
        w *= sigma * tr / mean(sigma * tr).max(1e-30);
        p += d * t;
        let z = 1.0 - 2.0 * rng.next();
        let phi = TAU * rng.next();
        let r = (1.0 - z * z).max(0.0).sqrt();
        d = Vec3::new(r * phi.cos(), r * phi.sin(), z);
    }
    None
}

/// Power-heuristic MIS weight of a strategy with density `a` against one with density `b`.
fn mis(a: f32, b: f32) -> f32 {
    let (a2, b2) = (a * a, b * b);
    if a2 + b2 > 0.0 { a2 / (a2 + b2) } else { 0.0 }
}

/// Clamp rare, very bright indirect contributions (fireflies) to keep progressive renders calm.
fn clamp_firefly(c: Vec3) -> Vec3 {
    let lum = luminance(c);
    if lum > 32.0 { c * (32.0 / lum) } else { c }
}

/// Next-event estimation toward the environment from `pos` (geometric normal `geo`): `eval`
/// gives the BRDF × cos toward a direction and the density of the path's own sampling of it.
fn next_event(scene: &TraceScene, pos: Vec3, geo: Vec3, thr: Vec3, rng: &mut Rng, eval: impl Fn(Vec3) -> (Vec3, f32)) -> Vec3 {
    let Some((wl, pdf_env)) = scene.sample_env(rng) else { return Vec3::ZERO };
    if wl.dot(geo) <= 0.0 {
        return Vec3::ZERO;
    }
    let (f, pdf_brdf) = eval(wl);
    if f.max_element() <= 0.0 || scene.bvh.occluded(pos + geo * scene.eps, wl, f32::MAX) {
        return Vec3::ZERO;
    }
    clamp_firefly(thr * f * scene.env(wl) * (mis(pdf_env, pdf_brdf) / pdf_env))
}

/// Radiance along a camera ray; None when it misses the mesh. Environment light is gathered
/// both by sampling it directly (next-event estimation) and by BRDF sampling, combined with MIS.
fn trace(scene: &TraceScene, mut o: Vec3, mut d: Vec3, settings: &TraceSettings, rng: &mut Rng) -> Option<Sample> {
    let origin = o;
    let mut first: Option<(Vec3, Vec3, f32)> = None;
    let mut thr = Vec3::ONE;
    let mut l = Vec3::ZERO;
    let mut bounce = 0u32;
    // Density of the BRDF sample that produced `d` (None for the camera ray).
    let mut brdf_pdf: Option<f32> = None;
    // Cutout surfaces don't count as bounces; cap them so a stack of cards can't loop forever.
    for _ in 0..settings.bounces + 24 {
        let Some(hit) = scene.bvh.closest(o, d, f32::MAX) else {
            let pdf = brdf_pdf?;
            let w = if scene.light.is_some() { mis(pdf, scene.env_pdf(d)) } else { 1.0 };
            l += clamp_firefly(thr * scene.env(d) * w);
            break;
        };
        let Some(s) = surface(scene, hit.tri, hit.u, hit.v, o, d, hit.t) else { break };
        if rng.next() > s.opacity {
            o = s.pos - s.geo * scene.eps;
            continue;
        }
        if first.is_none() {
            first = Some(((s.diffuse + s.f0).min(Vec3::ONE), s.n, (s.pos - origin).length()));
        }
        l += thr * s.emissive;
        if bounce >= settings.bounces {
            break;
        }
        // Transmission is picked by its weight (a glass event, without next-event estimation);
        // otherwise the surface shades as if it had none.
        let glass = s.transmission > 0.0 && rng.next() < s.transmission;
        let bsdf = Bsdf::new(&s, -d);
        // The BRDF is sampled with probability 1 − p_walk (densities below include that).
        let keep = 1.0 - bsdf.p_walk;
        if !glass {
            l += next_event(scene, s.pos, s.geo, thr, rng, |wl| {
                let (f, pdf) = bsdf.eval(wl);
                (f, pdf * keep)
            });
        }
        let (wi, geo) = if glass {
            let Some((wi, w)) = dielectric(&s, -d, scene.ior, rng) else { break };
            thr *= w;
            brdf_pdf = Some(UNSAMPLED_PDF);
            o = s.pos;
            (wi, s.geo)
        } else if bsdf.p_walk > 0.0 && rng.next() < bsdf.p_walk {
            let Some((exit, out, w)) = random_walk(scene, &s, rng) else { break };
            thr *= bsdf.walk_albedo * w / bsdf.p_walk;
            // The light leaves the exit point like a Lambertian emitter.
            l += next_event(scene, exit, out, thr, rng, |wl| {
                let c = wl.dot(out).max(0.0) / PI;
                (Vec3::splat(c), c)
            });
            let wi = cosine_around(out, rng);
            brdf_pdf = Some(wi.dot(out).max(1e-6) / PI);
            o = exit;
            (wi, out)
        } else {
            let Some(wi) = bsdf.sample(rng) else { break };
            let (f, pdf) = bsdf.eval(wi);
            if pdf <= 0.0 {
                break;
            }
            thr *= f / (pdf * keep);
            brdf_pdf = Some(pdf * keep);
            o = s.pos;
            (wi, s.geo)
        };
        if bounce >= 2 {
            let p = thr.max_element().clamp(0.05, 0.95);
            if rng.next() > p {
                break;
            }
            thr /= p;
        }
        let side = if wi.dot(geo) >= 0.0 { 1.0 } else { -1.0 };
        o += geo * scene.eps * side;
        d = wi;
        bounce += 1;
    }
    let (albedo, normal, depth) = first?;
    Some(Sample { radiance: if l.is_finite() { l } else { Vec3::ZERO }, albedo, normal, depth })
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_geom::primitives;

    fn scene(env: &str) -> TraceScene {
        let mesh = Arc::new(primitives::sphere(1.0, 32, 16));
        let bvh = Arc::new(Bvh::new(&mesh));
        let env = Arc::new(crate::ibl::environment(env, 128));
        TraceScene::new(mesh, bvh, Vec::new(), env, 0.0)
    }

    fn camera() -> OrbitCamera {
        let mut cam = OrbitCamera::default();
        cam.target = Vec3::ZERO;
        cam.distance = 4.0;
        cam
    }

    #[test]
    fn renders_the_mesh_over_the_backdrop_and_converges() {
        let s = scene("Studio Soft");
        let mut tr = Tracer::new(48, 32);
        for _ in 0..8 {
            tr.pass(&s, &camera(), &TraceSettings::default());
        }
        assert_eq!(tr.samples, 8);
        let d = TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.0, denoise: false };
        let img = tr.rgba8(&d);
        assert_eq!(img.len(), 48 * 32 * 4);
        let px = |x: usize, y: usize| &img[(y * 48 + x) * 4..(y * 48 + x) * 4 + 3];
        // The sphere (default material, lit by the studio) is brighter than the dark backdrop corner.
        let center: u32 = px(24, 16).iter().map(|v| *v as u32).sum();
        let corner: u32 = px(0, 0).iter().map(|v| *v as u32).sum();
        assert!(center > corner + 60, "center {center} corner {corner}");
        assert!(tr.acc.iter().all(|p| p.hit.is_finite() && p.env.is_finite()));
    }

    #[test]
    fn environment_shows_behind_the_mesh_with_opacity() {
        let s = scene("Overcast");
        let mut tr = Tracer::new(16, 8);
        tr.pass(&s, &camera(), &TraceSettings::default());
        let corner = |opacity: f32| tr.rgba8(&TraceDisplay { exposure: 1.0, aces: true, env_opacity: opacity, denoise: false })[0];
        assert!(corner(1.0) > corner(0.0) + 40);
    }

    #[test]
    fn light_sampling_matches_brdf_sampling_on_average() {
        let lit = scene("Studio Soft");
        let unlit = TraceScene { light: None, ..lit.clone() };
        let mean = |s: &TraceScene| {
            let mut rng = Rng::new(9, 9, 9);
            let n = 40_000;
            let mut sum = Vec3::ZERO;
            for _ in 0..n {
                sum += trace(s, Vec3::new(0.3, 0.4, 4.0), Vec3::new(-0.07, -0.1, -1.0).normalize(), &TraceSettings { bounces: 3 }, &mut rng)
                    .map_or(Vec3::ZERO, |s| s.radiance);
            }
            sum / n as f32
        };
        let (a, b) = (mean(&lit), mean(&unlit));
        assert!((a - b).length() < 0.05 * b.length().max(0.05), "with light sampling {a}, without {b}");
    }

    #[test]
    fn passes_split_into_row_slices_match_whole_passes() {
        let s = scene("Studio Soft");
        let (mut whole, mut sliced) = (Tracer::new(40, 30), Tracer::new(40, 30));
        let d = TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.0, denoise: false };
        for _ in 0..2 {
            whole.pass(&s, &camera(), &TraceSettings::default());
            let mut done = false;
            while !done {
                done = sliced.pass_rows(&s, &camera(), &TraceSettings::default(), 7);
            }
        }
        assert_eq!(sliced.samples, 2);
        assert_eq!(whole.rgba8(&d), sliced.rgba8(&d));
    }

    #[test]
    fn denoising_lowers_noise_and_keeps_the_mean() {
        let s = scene("Studio Contrast");
        let mut tr = Tracer::new(64, 48);
        for _ in 0..4 {
            tr.pass(&s, &camera(), &TraceSettings::default());
        }
        let raw = tr.rgba8(&TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.0, denoise: false });
        let clean = tr.rgba8(&TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.0, denoise: true });
        // Neighbour differences inside the ball (rows 18..30, columns 26..38) drop; the mean stays.
        let stats = |img: &[u8]| {
            let px = |x: usize, y: usize| img[(y * 64 + x) * 4] as f32;
            let (mut diff, mut sum, mut n) = (0.0, 0.0, 0.0);
            for y in 18..30 {
                for x in 26..38 {
                    diff += (px(x, y) - px(x + 1, y)).abs();
                    sum += px(x, y);
                    n += 1.0;
                }
            }
            (diff / n, sum / n)
        };
        let ((d_raw, m_raw), (d_clean, m_clean)) = (stats(&raw), stats(&clean));
        assert!(d_clean < d_raw * 0.6, "noise {d_raw} -> {d_clean}");
        assert!((m_raw - m_clean).abs() < 0.08 * m_raw.max(1.0), "mean {m_raw} -> {m_clean}");
    }

    type Values = Vec<(Channel, [f32; 3])>;

    /// A sphere whose texture set holds the uniform channel `values`.
    fn material_scene(env: Arc<EnvImage>, values: &[(Channel, [f32; 3])]) -> TraceScene {
        let mesh = Arc::new(primitives::sphere(1.0, 32, 16));
        let bvh = Arc::new(Bvh::new(&mesh));
        let chans: Vec<Channel> = values.iter().map(|(c, _)| *c).collect();
        let mut composite = Composite::new(&chans, 64);
        for (c, v) in values {
            let raw: Vec<u16> = v[..c.components()].iter().map(|x| texturecraft_texel::to_u16(*x)).collect();
            let img = composite.channels.get_mut(c).unwrap();
            for y in 0..64 {
                for x in 0..64 {
                    img.set(x, y, &raw);
                }
            }
        }
        TraceScene::new(mesh, bvh, vec![TraceSet { material: 0, composite, height_strength: 0.0 }], env, 0.0)
    }

    /// Mean radiance of parallel rays over the whole disc of the sphere (grazing hits included)
    /// in a uniform white environment.
    fn furnace(values: &[(Channel, [f32; 3])]) -> Vec3 {
        let env = Arc::new(EnvImage { width: 8, height: 4, data: vec![[1.0; 3]; 32] });
        let s = material_scene(env, values);
        let mut rng = Rng::new(4, 5, 6);
        let (mut sum, mut n) = (Vec3::ZERO, 0);
        while n < 6000 {
            let (x, y) = (rng.next() * 2.0 - 1.0, rng.next() * 2.0 - 1.0);
            if x * x + y * y > 0.97 {
                continue;
            }
            sum += trace(&s, Vec3::new(x, y, 4.0), Vec3::NEG_Z, &TraceSettings { bounces: 8 }, &mut rng).map_or(Vec3::ZERO, |s| s.radiance);
            n += 1;
        }
        sum / n as f32
    }

    const WHITE: [(Channel, [f32; 3]); 3] = [(Channel::BaseColor, [1.0; 3]), (Channel::Metallic, [0.0; 3]), (Channel::Roughness, [0.5; 3])];

    #[test]
    fn white_furnace_with_each_lobe_at_full_weight_stays_bounded() {
        use Channel::*;
        let lobes: [(&str, Values); 5] = [
            ("coat", vec![(CoatWeight, [1.0; 3]), (CoatRoughness, [0.2; 3])]),
            ("sheen", vec![(SheenWeight, [1.0; 3]), (SheenColor, [1.0; 3]), (SheenRoughness, [0.5; 3])]),
            ("anisotropic metal", vec![(AnisotropyLevel, [1.0; 3]), (AnisotropyAngle, [0.3; 3]), (Metallic, [1.0; 3]), (Roughness, [0.4; 3])]),
            ("scattering", vec![(ScatterWeight, [1.0; 3])]),
            ("everything", vec![(CoatWeight, [1.0; 3]), (SheenWeight, [1.0; 3]), (AnisotropyLevel, [1.0; 3]), (ScatterWeight, [1.0; 3])]),
        ];
        for (name, extra) in lobes {
            // Later values of a channel win (the anisotropic metal's metallic and roughness).
            let mut values: Values = WHITE.into_iter().filter(|(c, _)| !extra.iter().any(|(e, _)| e == c)).collect();
            values.extend(extra);
            let avg = furnace(&values);
            assert!(avg.max_element() <= 1.05 && avg.min_element() > 0.5, "{name}: {avg}");
        }
    }

    #[test]
    fn lobes_sample_and_evaluate_consistently() {
        // Light sampling and BRDF sampling (MIS) against BRDF sampling alone: equal means only
        // when every lobe's density matches what it samples.
        use Channel::*;
        let env = Arc::new(crate::ibl::environment("Studio Soft", 128));
        let lit = material_scene(
            env,
            &[
                (BaseColor, [0.8, 0.5, 0.3]),
                (Roughness, [0.35; 3]),
                (CoatWeight, [0.8; 3]),
                (CoatRoughness, [0.15; 3]),
                (SheenWeight, [0.6; 3]),
                (SheenColor, [0.4, 0.6, 1.0]),
                (AnisotropyLevel, [0.8; 3]),
                (AnisotropyAngle, [0.2; 3]),
                (ScatterWeight, [0.7; 3]),
            ],
        );
        let unlit = TraceScene { light: None, ..lit.clone() };
        let mean = |s: &TraceScene| {
            let mut rng = Rng::new(9, 9, 9);
            let n = 100_000;
            let mut sum = Vec3::ZERO;
            for _ in 0..n {
                sum += trace(s, Vec3::new(0.3, 0.4, 4.0), Vec3::new(-0.07, -0.1, -1.0).normalize(), &TraceSettings { bounces: 3 }, &mut rng)
                    .map_or(Vec3::ZERO, |s| s.radiance);
            }
            sum / n as f32
        };
        let (a, b) = (mean(&lit), mean(&unlit));
        assert!((a - b).length() < 0.05 * b.length().max(0.05), "with light sampling {a}, without {b}");
    }

    #[test]
    fn sheen_albedo_fit_matches_the_integral() {
        for r in [0.25f32, 0.5, 0.75, 1.0] {
            let a = r * r;
            for nov in [0.2f32, 0.5, 0.8, 1.0] {
                let v = Vec3::new((1.0 - nov * nov).sqrt(), 0.0, nov);
                let (nt, np) = (128, 256);
                let mut e = 0.0;
                for i in 0..nt {
                    let th = (i as f32 + 0.5) / nt as f32 * PI / 2.0;
                    for j in 0..np {
                        let ph = (j as f32 + 0.5) / np as f32 * TAU;
                        let l = Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                        let h = (l + v).normalize();
                        e += d_charlie(a, h.z) * sheen_vis(nov, l.z, a) * l.z * th.sin();
                    }
                }
                e *= (PI / 2.0 / nt as f32) * (TAU / np as f32);
                let fit = sheen_albedo(nov, r);
                assert!((fit - e).abs() < 0.04, "r {r} cos {nov}: integral {e}, fit {fit}");
            }
        }
    }

    #[test]
    fn white_furnace_keeps_energy_bounded() {
        // A uniform white environment: a white diffuse sphere must not exceed the environment.
        let mesh = Arc::new(primitives::sphere(1.0, 24, 12));
        let bvh = Arc::new(Bvh::new(&mesh));
        let env = Arc::new(EnvImage { width: 8, height: 4, data: vec![[1.0; 3]; 32] });
        let s = TraceScene::new(mesh, bvh, Vec::new(), env, 0.0);
        let mut rng = Rng::new(1, 2, 3);
        let mut sum = Vec3::ZERO;
        let n = 4000;
        for _ in 0..n {
            sum += trace(&s, Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z, &TraceSettings { bounces: 8 }, &mut rng).map_or(Vec3::ZERO, |s| s.radiance);
        }
        let avg = sum / n as f32;
        assert!(avg.max_element() <= 1.05 && avg.min_element() > 0.3, "{avg}");
    }

    #[test]
    fn glass_white_furnace_stays_bounded() {
        for rough in [0.0, 0.3, 0.7] {
            let avg = furnace(&[(Channel::BaseColor, [1.0; 3]), (Channel::Roughness, [rough; 3]), (Channel::Transmission, [1.0; 3])]);
            assert!(avg.max_element() <= 1.05 && avg.min_element() > 0.5, "roughness {rough}: {avg}");
        }
    }

    /// Mean radiance of the center ray from +z at the unit sphere, with an environment that is
    /// bright only behind the sphere (toward -z).
    fn behind_mean(values: &[(Channel, [f32; 3])]) -> Vec3 {
        let (w, h) = (64u32, 32u32);
        let data = (0..w * h)
            .map(|i| if crate::ibl::dir_of(((i % w) as f32 + 0.5) / w as f32, ((i / w) as f32 + 0.5) / h as f32).z < -0.7 { [5.0; 3] } else { [0.01; 3] })
            .collect();
        let s = material_scene(Arc::new(EnvImage { width: w, height: h, data }), values);
        let mut rng = Rng::new(3, 1, 4);
        let n = 2000;
        (0..n).map(|_| trace(&s, Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z, &TraceSettings::default(), &mut rng).map_or(Vec3::ZERO, |s| s.radiance)).sum::<Vec3>()
            / n as f32
    }

    #[test]
    fn transmission_shows_the_environment_behind_the_mesh() {
        let white = [(Channel::BaseColor, [1.0; 3]), (Channel::Roughness, [0.05; 3])];
        let opaque = behind_mean(&white);
        let glass = behind_mean(&[white[0], white[1], (Channel::Transmission, [1.0; 3])]);
        assert!(glass.x > 2.0 && glass.x > 10.0 * opaque.x, "through the glass {glass}, opaque {opaque}");
        // The base color tints the light going through.
        let tinted = behind_mean(&[(Channel::BaseColor, [1.0, 0.2, 0.2]), white[1], (Channel::Transmission, [1.0; 3])]);
        assert!(tinted.x > 3.0 * tinted.y, "{tinted}");
    }

    #[test]
    fn thin_lens_blurs_away_from_the_focus_plane() {
        // Pixels between the sphere and the backdrop: how soft the silhouette is.
        let soft_edge = |lens: Option<[f32; 2]>| {
            let s = TraceScene { lens, ..scene("Studio Soft") };
            let mut tr = Tracer::new(64, 48);
            for _ in 0..24 {
                tr.pass(&s, &camera(), &TraceSettings { bounces: 1 });
            }
            tr.acc.iter().filter(|p| p.hits > 2.0 && p.hits < tr.samples as f32 - 2.0).count()
        };
        let (pinhole, focused, blurred) = (soft_edge(None), soft_edge(Some([4.0 - 0.25, 0.1])), soft_edge(Some([20.0, 0.1])));
        assert!(blurred > 3 * pinhole.max(10), "out of focus: {pinhole} -> {blurred} edge pixels");
        assert!(focused < pinhole + pinhole / 2 + 10, "the silhouette (about the focus depth) stays sharp: {pinhole} -> {focused}");
    }
}
