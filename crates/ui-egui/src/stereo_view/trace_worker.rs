//! One cancellable CPU pair worker; a replacement waits for the retiring worker.
use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;
use texturecraft_render::trace_pair::{CpuTracePair, TracePairFrame};

pub(super) struct Completed {
    pub pixels: [Vec<u8>; 2],
    pub samples: [u32; 2],
}

pub(super) struct Worker {
    pub key: String,
    cancel: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<Result<Completed, String>>>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn start(key: String, frame: TracePairFrame, ctx: egui::Context) -> Result<Self, String> {
        let cancel = Arc::new(AtomicBool::new(false));
        let latest = Arc::new(Mutex::new(None));
        let stop = cancel.clone();
        let output = latest.clone();
        let handle = std::thread::Builder::new()
            .name("TextureCraft stereo trace".into())
            .spawn(move || {
                let run = || -> Result<(), String> {
                    let max_samples = frame.max_samples.max(1);
                    let mut pair = CpuTracePair::new(frame)?;
                    while pair.samples()[0] < max_samples {
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        if pair.advance_rows(4)? {
                            let completed = Completed { samples: pair.samples(), pixels: pair.rgba8()? };
                            if stop.load(Ordering::Acquire) {
                                return Ok(());
                            }
                            *output.lock().unwrap_or_else(PoisonError::into_inner) = Some(Ok(completed));
                            ctx.request_repaint();
                        }
                    }
                    Ok(())
                };
                if let Err(error) = run() {
                    *output.lock().unwrap_or_else(PoisonError::into_inner) = Some(Err(error));
                    ctx.request_repaint();
                }
            })
            .map_err(|e| format!("could not start the stereo trace worker: {e}"))?;
        Ok(Self { key, cancel, latest, handle: Some(handle) })
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
    pub fn finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }
    pub fn take(&self) -> Option<Result<Completed, String>> {
        self.latest.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
    pub fn join(&mut self) -> Result<(), String> {
        match self.handle.take() {
            Some(handle) => handle.join().map_err(|_| "the stereo trace worker stopped unexpectedly".into()),
            None => Ok(()),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel();
        // A row slice bounds cancellation latency; never leave an unowned worker behind.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_engine::geom::{Bvh, OrbitCamera, primitives};
    use texturecraft_render::{
        stereo::Budget,
        trace::{TraceDisplay, TraceScene, TraceSettings},
    };

    fn frame(samples: u32) -> TracePairFrame {
        let mesh = Arc::new(primitives::sphere(1.0, 16, 8));
        let bvh = Arc::new(Bvh::new(&mesh));
        let scene = Arc::new(TraceScene::new(mesh, bvh, vec![], Arc::new(texturecraft_render::ibl::environment("Studio Contrast", 32)), 0.0));
        TracePairFrame {
            scene,
            camera: OrbitCamera { distance: 4.0, ..Default::default() },
            size: [32, 24],
            budget: Budget { sunk: 2.0, popped: 1.0 },
            settings: TraceSettings { bounces: 1 },
            display: TraceDisplay { exposure: 1.0, aces: false, env_opacity: 1.0, denoise: false },
            max_samples: samples,
            epoch: "one worker-owned source".into(),
        }
    }

    #[test]
    fn completed_worker_releases_its_world_and_returns_one_equal_sample_pair() {
        let frame = frame(2);
        let scene = frame.scene.clone();
        let mut job = Worker::start(frame.key(), frame, egui::Context::default()).unwrap();
        job.join().unwrap();
        assert!(job.finished());
        let completed = job.take().unwrap().unwrap();
        assert_eq!(completed.samples, [2, 2]);
        for pixels in completed.pixels {
            assert_eq!(pixels.len(), 32 * 24 * 4);
        }
        assert_eq!(Arc::strong_count(&scene), 1, "a finished worker must not retain an obsolete scene");
        assert!(job.take().is_none(), "a complete pair is consumed once");
    }

    #[test]
    fn cancellation_joins_a_worker_that_has_actually_traced_a_pair() {
        let frame = frame(u32::MAX);
        let scene = frame.scene.clone();
        let mut job = Worker::start(frame.key(), frame, egui::Context::default()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(completed) = job.take() {
                let completed = completed.unwrap();
                assert!(completed.samples[0] > 0);
                assert_eq!(completed.samples[0], completed.samples[1]);
                break;
            }
            assert!(std::time::Instant::now() < deadline, "the worker did not complete its first pair");
            std::thread::yield_now();
        }
        job.cancel();
        job.join().unwrap();
        assert!(job.finished() && job.cancelled());
        assert_eq!(Arc::strong_count(&scene), 1, "joining acknowledges release before a replacement can start");
        if let Some(completed) = job.take() {
            let completed = completed.unwrap();
            assert_eq!(completed.samples[0], completed.samples[1]);
        }
    }

    #[test]
    fn a_rejected_camera_is_reported_by_the_worker_instead_of_stalling() {
        let mut frame = frame(2);
        frame.size[0] = 0;
        let mut job = Worker::start(frame.key(), frame, egui::Context::default()).unwrap();
        job.join().unwrap();
        let error = job.take().unwrap().err().expect("invalid dimensions must be reported");
        assert!(error.contains("invalid"), "{error}");
    }
}
