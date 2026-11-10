//! Asynchronous layer-panel thumbnails.
//!
//! The panel asks for small previews of layer contents, masks and effect stages
//! ([`ThumbTarget`]). The work never runs on the UI thread and never touches the main composite:
//!
//! * A **snapshot** ([`ThumbSnapshot`]) is an immutable copy of what the compositor reads (the
//!   project, copy-on-write pixel tiles, baked maps, images and the mesh) plus a 64 px G-buffer.
//!   It is rebuilt only when the document really changed, at most every [`DEBOUNCE`] seconds
//!   (a slider drag renders at that pace, not once per frame).
//! * A **worker thread** evaluates one target at a time from a snapshot with
//!   [`texturecraft_compose::thumb::render`]. On wasm32 (no threads) one job runs per
//!   [`ThumbService::poll`] instead.
//! * Results are kept in a small LRU **cache** ([`CACHE_ENTRIES`]). A result computed from an
//!   older snapshot than the current one is dropped, so a slow job can never overwrite a newer
//!   thumbnail. An entry that is out of date keeps showing until its replacement arrives.
//! * Only targets requested during the current frame are rendered: a row that scrolled away
//!   before its turn is dropped from the queue.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use texturecraft_compose::{BakedMaps, ImageStore, Inputs, PixelStore, THUMB_TEXELS, ThumbImage, ThumbTarget, thumb};
use texturecraft_core::Project;
use texturecraft_geom::glam::Vec3;
use texturecraft_geom::{Mesh, UvGBuffer};

use crate::Session;

/// Thumbnails kept.
pub const CACHE_ENTRIES: usize = 96;
/// Seconds between checks for a changed document while it keeps changing.
pub const DEBOUNCE: f64 = 0.12;
/// Jobs handed to the worker at once (the rest wait here, where they can still be dropped).
const IN_FLIGHT: usize = 2;

/// An immutable copy of everything the thumbnails of one texture set read.
pub struct ThumbSnapshot {
    ts: usize,
    mesh_id: u64,
    project: Project,
    pixels: PixelStore,
    baked: BakedMaps,
    images: ImageStore,
    mesh: Arc<Mesh>,
    gbuf: Arc<UvGBuffer>,
    bounds: (Vec3, Vec3),
}

impl ThumbSnapshot {
    /// Snapshot texture set `ts` of `s`, with the G-buffer from `gbufs` (built on a miss).
    pub fn new(s: &Session, ts: usize, gbufs: &mut GbufCache) -> Option<ThumbSnapshot> {
        let material = s.doc.project.texture_set(ts).ok()?.material;
        let gbuf = gbufs.get(&s.mesh, s.mesh_id, material);
        Some(ThumbSnapshot {
            ts,
            mesh_id: s.mesh_id,
            project: s.doc.project.clone(),
            pixels: s.doc.pixels.clone(),
            baked: s.doc.baked.get(&ts).cloned().unwrap_or_default(),
            images: s.images.clone(),
            mesh: s.mesh.clone(),
            gbuf,
            bounds: s.mesh.bounds(),
        })
    }

    /// Whether `session` would give the same thumbnails as this snapshot.
    fn matches(&self, session: &Session, texture_set: usize) -> bool {
        let same_pixels = self.pixels.layers.len() == session.doc.pixels.layers.len()
            && self.pixels.masks.len() == session.doc.pixels.masks.len()
            && self.pixels.layers.iter().all(|(key, pixels)| session.doc.pixels.layers.get(key) == Some(pixels))
            && self.pixels.masks.iter().all(|(key, pixels)| session.doc.pixels.masks.get(key) == Some(pixels));
        let empty = BakedMaps::default();
        let baked = session.doc.baked.get(&texture_set).unwrap_or(&empty);

        // Copy-on-write maps change handles on edits; float equality would scan entire bakes
        let same_baked = self.baked.resolution == baked.resolution
            && self.baked.maps.len() == baked.maps.len()
            && self.baked.maps.iter().all(|(map, pixels)| baked.maps.get(map).is_some_and(|current| Arc::ptr_eq(pixels, current)));

        self.ts == texture_set
            && self.mesh_id == session.mesh_id
            && self.project == session.doc.project
            && same_pixels
            && same_baked
            && self.images.len() == session.images.len()
            && self.images.iter().all(|(key, image)| session.images.get(key).is_some_and(|current| Arc::ptr_eq(current, image)))
    }

    /// Render `target`: the work of one job.
    pub fn render(&self, target: ThumbTarget) -> Option<ThumbImage> {
        let inp = Inputs {
            project: &self.project,
            ts: self.ts,
            mesh: &self.mesh,
            gbuf: &self.gbuf,
            pixels: &self.pixels,
            baked: &self.baked,
            block_scales: self.project.texture_set(self.ts).map(|set| set.block_scales()).unwrap_or_default(),
            preview: None,
            images: &self.images,
            bounds: self.bounds,
        };
        thumb::render(&inp, target)
    }

    /// Whether the layer or effect `target` shows still exists in the snapshot's project.
    fn has(&self, target: ThumbTarget) -> bool {
        has_target(&self.project, target)
    }
}

fn has_target(p: &Project, target: ThumbTarget) -> bool {
    match target {
        ThumbTarget::Content { layer, .. } | ThumbTarget::Mask { layer } => p.layers.contains_key(&layer),
        ThumbTarget::Stage { effect, .. } => p.effect(effect).is_some(),
    }
}

/// The 64 px G-buffers built so far, per mesh and material (rasterizing is the one part of a
/// snapshot that is not just copying handles).
#[derive(Default)]
pub struct GbufCache {
    map: HashMap<(u64, usize), Arc<UvGBuffer>>,
}

impl GbufCache {
    fn get(&mut self, mesh: &Mesh, mesh_id: u64, material: usize) -> Arc<UvGBuffer> {
        if self.map.len() > 16 {
            self.map.clear();
        }
        self.map.entry((mesh_id, material)).or_insert_with(|| Arc::new(UvGBuffer::new(mesh, material, THUMB_TEXELS, THUMB_TEXELS))).clone()
    }
}

struct Entry {
    /// None: the target has nothing to show (a layer without that channel).
    image: Option<Arc<ThumbImage>>,
    /// Snapshot generation it was rendered from.
    generation: u64,
    /// Changes whenever `image` does (views key their GPU textures on it).
    version: u64,
    used: u64,
}

/// What the panel draws for one target.
#[derive(Clone)]
pub struct ThumbView {
    pub image: Option<Arc<ThumbImage>>,
    /// Identifies `image`; changes when a new picture arrives.
    pub version: u64,
    /// A newer picture is on its way.
    pub pending: bool,
}

struct Job {
    snapshot: Arc<ThumbSnapshot>,
    generation: u64,
    target: ThumbTarget,
}

struct Done {
    generation: u64,
    target: ThumbTarget,
    image: Option<ThumbImage>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Worker {
    tx: std::sync::mpsc::Sender<Job>,
    rx: std::sync::mpsc::Receiver<Done>,
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_worker() -> Option<Worker> {
    let (tx, jobs) = std::sync::mpsc::channel::<Job>();
    let (done_tx, rx) = std::sync::mpsc::channel::<Done>();
    std::thread::Builder::new()
        .name("layer-thumbs".into())
        .spawn(move || {
            while let Ok(job) = jobs.recv() {
                let image = job.snapshot.render(job.target);
                if done_tx.send(Done { generation: job.generation, target: job.target, image }).is_err() {
                    break;
                }
            }
        })
        .ok()?;
    Some(Worker { tx, rx })
}

/// The thumbnail cache and its worker. One lives in the UI; see the module documentation.
pub struct ThumbService {
    snapshot: Option<Arc<ThumbSnapshot>>,
    /// Bumps whenever the snapshot's content changes; entries and jobs carry it.
    generation: u64,
    /// The session revision the snapshot was last compared with.
    seen: Option<u64>,
    last_check: f64,
    cache: HashMap<ThumbTarget, Entry>,
    tick: u64,
    versions: u64,
    queue: VecDeque<ThumbTarget>,
    in_flight: HashSet<(u64, ThumbTarget)>,
    wanted: HashSet<ThumbTarget>,
    gbufs: GbufCache,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<Worker>,
    #[cfg(not(target_arch = "wasm32"))]
    spawned: bool,
    /// Jobs run inline (wasm32, or no worker thread could start).
    inline: VecDeque<Job>,
    /// Pictures computed (tests, benchmarks).
    pub rendered: u64,
}

impl Default for ThumbService {
    fn default() -> Self {
        ThumbService {
            snapshot: None,
            generation: 0,
            seen: None,
            last_check: f64::NEG_INFINITY,
            cache: HashMap::new(),
            tick: 0,
            versions: 0,
            queue: VecDeque::new(),
            in_flight: HashSet::new(),
            wanted: HashSet::new(),
            gbufs: GbufCache::default(),
            #[cfg(not(target_arch = "wasm32"))]
            worker: None,
            #[cfg(not(target_arch = "wasm32"))]
            spawned: false,
            inline: VecDeque::new(),
            rendered: 0,
        }
    }
}

impl ThumbService {
    pub fn new() -> ThumbService {
        ThumbService::default()
    }

    /// Start of a frame: forget what the last frame asked for and notice document changes.
    pub fn begin_frame(&mut self, session: &Session, now: f64) {
        self.wanted.clear();
        let texture_set = session.ts();
        if self.seen == Some(session.revision)
            && self.snapshot.as_ref().is_some_and(|snapshot| snapshot.ts == texture_set && snapshot.mesh_id == session.mesh_id)
        {
            return;
        }

        // Compare at the debounce pace while the document keeps changing
        if self.snapshot.is_some() && now - self.last_check < DEBOUNCE {
            return;
        }

        self.last_check = now;
        if self.snapshot.as_ref().is_some_and(|snapshot| snapshot.matches(session, texture_set)) {
            self.seen = Some(session.revision);
            return;
        }

        self.snapshot = ThumbSnapshot::new(session, texture_set, &mut self.gbufs).map(Arc::new);
        self.seen = Some(session.revision);
        self.generation = self.generation.wrapping_add(1);

        // Keep dispatched jobs counted until they finish, even when their snapshot is stale
        self.queue.clear();
        for job in self.inline.drain(..) {
            self.in_flight.remove(&(job.generation, job.target));
        }

        if let Some(snapshot) = &self.snapshot {
            self.cache.retain(|target, _| snapshot.has(*target));
        } else {
            self.cache.clear();
        }
    }

    /// Whether the displayed pictures wait for a document comparison that is still debounced.
    fn checked(&self, session: &Session) -> bool {
        self.seen == Some(session.revision)
    }

    /// The picture for `target` (call for the rows on screen only). Queues work when the
    /// picture is missing or out of date.
    pub fn get(&mut self, session: &Session, target: ThumbTarget) -> ThumbView {
        self.wanted.insert(target);
        self.tick = self.tick.wrapping_add(1);
        let fresh = self.cache.get(&target).is_some_and(|entry| entry.generation == self.generation);
        let checked = self.checked(session);
        if !fresh && checked && self.snapshot.is_some() && !self.in_flight.contains(&(self.generation, target)) && !self.queue.contains(&target) {
            self.queue.push_back(target);
        }

        match self.cache.get_mut(&target) {
            Some(entry) => {
                entry.used = self.tick;
                ThumbView { image: entry.image.clone(), version: entry.version, pending: !(fresh && checked) }
            }
            None => ThumbView { image: None, version: 0, pending: true },
        }
    }

    /// End of a frame: collect results, discard unrequested rows, and dispatch new work.
    /// Returns whether another frame is needed, including to display newly collected pictures.
    pub fn poll(&mut self, session: &Session) -> bool {
        let changed = self.collect();
        self.queue.retain(|target| self.wanted.contains(target));
        self.dispatch();
        self.trim();

        changed || !self.queue.is_empty() || !self.in_flight.is_empty() || !self.checked(session)
    }

    fn dispatch(&mut self) {
        let Some(snapshot) = self.snapshot.clone() else { return };

        while self.in_flight.len() < IN_FLIGHT {
            let Some(target) = self.queue.pop_front() else { break };
            let job = Job { snapshot: snapshot.clone(), generation: self.generation, target };
            self.in_flight.insert((job.generation, target));
            self.send(job);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn send(&mut self, job: Job) {
        if !self.spawned {
            self.spawned = true;
            self.worker = spawn_worker();
        }

        match &self.worker {
            Some(worker) => {
                if let Err(error) = worker.tx.send(job) {
                    // A stopped worker cannot complete any of its outstanding jobs
                    self.worker = None;
                    self.in_flight.clear();
                    self.in_flight.insert((error.0.generation, error.0.target));
                    self.inline.push_back(error.0);
                }
            }
            None => self.inline.push_back(job),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn send(&mut self, job: Job) {
        self.inline.push_back(job);
    }

    fn collect(&mut self) -> bool {
        let mut completed = Vec::new();
        let mut changed = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(worker) = &self.worker {
            loop {
                match worker.rx.try_recv() {
                    Ok(result) => completed.push(result),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        // Request another frame so wanted targets retry on the inline fallback
                        self.worker = None;
                        self.in_flight.clear();
                        changed = true;
                        break;
                    }
                }
            }
        }

        // Without a thread, bound each poll to one render
        if let Some(job) = self.inline.pop_front() {
            let image = job.snapshot.render(job.target);
            completed.push(Done { generation: job.generation, target: job.target, image });
        }

        for result in completed {
            self.rendered = self.rendered.wrapping_add(1);
            self.in_flight.remove(&(result.generation, result.target));

            // A picture of an older snapshot never replaces a newer one
            if result.generation != self.generation {
                continue;
            }

            if !self.snapshot.as_ref().is_some_and(|snapshot| snapshot.has(result.target)) {
                continue;
            }

            self.versions = self.versions.wrapping_add(1);
            let entry = Entry { image: result.image.map(Arc::new), generation: result.generation, version: self.versions, used: self.tick };
            self.cache.insert(result.target, entry);
            changed = true;
        }

        changed
    }

    fn trim(&mut self) {
        while self.cache.len() > CACHE_ENTRIES {
            let Some(oldest) = self.cache.iter().filter(|(t, _)| !self.wanted.contains(t)).min_by_key(|(_, e)| e.used).map(|(t, _)| *t) else { break };
            self.cache.remove(&oldest);
        }
    }

    /// Entries held (tests).
    pub fn cached(&self) -> usize {
        self.cache.len()
    }

    /// Whether `target` has a picture of the current snapshot (tests).
    pub fn is_fresh(&self, target: ThumbTarget) -> bool {
        self.cache.get(&target).is_some_and(|e| e.generation == self.generation)
    }

    /// Whether any work is queued or running (tests).
    pub fn busy(&self) -> bool {
        !self.queue.is_empty() || !self.in_flight.is_empty() || !self.inline.is_empty()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use texturecraft_core::Channel;
    use texturecraft_geom::primitives;

    fn test_session() -> crate::Result<Session> {
        Session::new_project("thumb", primitives::plane(1.0, 1), "builtin:plane", None, 128)
    }

    fn held_worker() -> (ThumbService, std::sync::mpsc::Receiver<Job>, std::sync::mpsc::Sender<Done>) {
        let (sender, jobs) = std::sync::mpsc::channel();
        let (results, receiver) = std::sync::mpsc::channel();
        let service = ThumbService { worker: Some(Worker { tx: sender, rx: receiver }), spawned: true, ..Default::default() };
        (service, jobs, results)
    }

    #[test]
    fn unchanged_baked_handles_match_even_with_non_finite_samples() {
        let mut session = test_session().unwrap();
        let baked = session.doc.baked.entry(0).or_default();
        baked.resolution = 1;
        baked.maps.insert(texturecraft_core::BakedMap::Height, Arc::new(vec![f32::NAN]));
        let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();

        assert!(snapshot.matches(&session, 0), "unchanged shared data must not invalidate every frame");

        let map = session.doc.baked.get_mut(&0).unwrap().maps.get_mut(&texturecraft_core::BakedMap::Height).unwrap();
        Arc::make_mut(map)[0] = 0.5;
        assert!(!snapshot.matches(&session, 0), "a copy-on-write edit must invalidate the snapshot");
    }

    #[test]
    #[ignore]
    fn benchmark_unchanged_4k_baked_snapshot() {
        let mut session = test_session().unwrap();
        let baked = session.doc.baked.entry(0).or_default();
        baked.resolution = 4096;
        baked.maps.insert(texturecraft_core::BakedMap::Normal, Arc::new(vec![0.5; 4096 * 4096 * 3]));
        baked.maps.insert(texturecraft_core::BakedMap::AmbientOcclusion, Arc::new(vec![0.5; 4096 * 4096]));
        let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();
        let start = std::time::Instant::now();

        for _ in 0..20 {
            assert!(std::hint::black_box(&snapshot).matches(std::hint::black_box(&session), 0));
        }

        eprintln!("20 unchanged snapshot comparisons, 4K normal + AO (256 MiB): {:?}", start.elapsed());
    }

    #[test]
    fn collecting_the_last_result_requests_a_frame_to_display_it() {
        let session = test_session().unwrap();
        let target = ThumbTarget::Content { layer: session.selected_layer().unwrap(), channel: Channel::BaseColor };
        let (mut service, jobs, results) = held_worker();
        service.begin_frame(&session, 0.0);
        service.get(&session, target);
        assert!(service.poll(&session));
        let job = jobs.try_recv().unwrap();
        results.send(Done { generation: job.generation, target, image: job.snapshot.render(target) }).unwrap();

        service.begin_frame(&session, 0.1);
        assert!(service.get(&session, target).image.is_none());
        let repaint = service.poll(&session);

        assert!(service.is_fresh(target));
        assert!(repaint, "the frame drew before collecting the final thumbnail");
        service.begin_frame(&session, 0.2);
        assert!(service.get(&session, target).image.is_some());
        assert!(!service.poll(&session), "an idle cache must stop repainting");
    }

    #[test]
    fn a_stopped_worker_retries_visible_targets_inline() {
        let session = test_session().unwrap();
        let target = ThumbTarget::Content { layer: session.selected_layer().unwrap(), channel: Channel::BaseColor };
        let (mut service, jobs, results) = held_worker();
        service.begin_frame(&session, 0.0);
        service.get(&session, target);
        service.poll(&session);

        // Simulate a worker that stops with a dispatched request still outstanding
        drop(jobs);
        drop(results);
        for frame in 1..=4 {
            service.begin_frame(&session, f64::from(frame));
            service.get(&session, target);
            service.poll(&session);
        }

        assert!(service.get(&session, target).image.is_some());
        assert!(!service.busy());
    }

    #[test]
    fn edits_keep_worker_jobs_and_retained_snapshots_bounded() {
        let mut session = test_session().unwrap();
        let layer = session.selected_layer().unwrap();
        let target = ThumbTarget::Content { layer, channel: Channel::BaseColor };
        let (mut service, jobs, results) = held_worker();
        let mut snapshots = Vec::new();

        // Hold the worker's receiver still while repeated edits each produce a new snapshot
        for revision in 0..2_000 {
            session.doc.pixels.layer_mut(layer, Channel::BaseColor, (128, 128)).set(0, 0, &[revision, 0, 0, u16::MAX]);
            session.revision += 1;
            service.begin_frame(&session, f64::from(revision));
            snapshots.push(Arc::downgrade(service.snapshot.as_ref().unwrap()));
            service.get(&session, target);
            service.poll(&session);
        }

        let queued: Vec<_> = jobs.try_iter().collect();
        let retained = snapshots.iter().filter(|snapshot| snapshot.strong_count() > 0).count();
        let retained_pixel_bytes: usize = snapshots.iter().filter_map(std::sync::Weak::upgrade).map(|snapshot| snapshot.pixels.bytes()).sum();
        eprintln!("2,000 edits: {} queued jobs, {retained} live snapshots, {retained_pixel_bytes} retained pixel bytes", queued.len());
        assert!(queued.len() <= IN_FLIGHT, "{} queued jobs bypassed the worker limit", queued.len());
        assert!(retained <= IN_FLIGHT + 1, "{retained} document snapshots retained across edits");
        assert!(retained_pixel_bytes <= (IN_FLIGHT + 1) * texturecraft_texel::TILE_AREA * 4 * std::mem::size_of::<u16>());

        // Finishing stale work frees a slot for the newest document, without caching old results
        for job in queued {
            results.send(Done { generation: job.generation, target: job.target, image: None }).unwrap();
        }

        assert_eq!(snapshots.iter().filter(|snapshot| snapshot.strong_count() > 0).count(), 1, "completed jobs release stale snapshots");
        assert!(service.poll(&session));
        assert!(!service.is_fresh(target));
        let latest = jobs.try_recv().unwrap();
        assert_eq!(latest.generation, service.generation);
        results.send(Done { generation: latest.generation, target, image: latest.snapshot.render(target) }).unwrap();
        drop(latest);
        assert!(service.poll(&session));
        assert!(service.is_fresh(target));

        let image = Arc::downgrade(service.cache.get(&target).unwrap().image.as_ref().unwrap());
        drop(service);

        assert!(snapshots.iter().all(|snapshot| snapshot.upgrade().is_none()), "dropping the cache releases its last snapshot");
        assert!(image.upgrade().is_none(), "dropping the cache releases the rendered image");
        assert!(jobs.try_recv().is_err());
    }
}
