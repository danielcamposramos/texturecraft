//! Path tool geometry: a smooth curve through control points and the antialiased coverage of
//! a closed curve (lasso fill). Works in any 2D space (screen pixels or texels).

use texturecraft_geom::glam::Vec2;

/// Most samples per curve segment.
const MAX_SEGMENT_SAMPLES: f32 = 1024.0;

/// A Catmull-Rom spline through `points`, sampled about every `step` units, starting on the first point (closed curves end there too)
pub fn spline(points: &[Vec2], closed: bool, step: f32) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = points.first().copied().into_iter().collect();
    for segment in segment_samples(points, closed, step) {
        out.extend(segment);
    }
    out
}

/// [`spline`]'s samples per segment: segment `s` holds those after point `s`, up to and including the next point
pub fn spline_segments(points: &[Vec2], closed: bool, step: f32) -> Vec<Vec<Vec2>> {
    segment_samples(points, closed, step).map(Iterator::collect).collect()
}

/// Stream samples so a flat curve does not allocate and retain every segment alongside its output
fn segment_samples(points: &[Vec2], closed: bool, step: f32) -> impl Iterator<Item = impl Iterator<Item = Vec2>> + '_ {
    let n = points.len();
    let at = move |i: isize| {
        let i = if closed { i.rem_euclid(n as isize) } else { i.clamp(0, n as isize - 1) };
        points.get(i as usize).copied().unwrap_or_default()
    };
    let step = if step.is_finite() && step > 1e-6 { step } else { 1.0 };
    let segments = if n < 2 {
        0
    } else if closed {
        n
    } else {
        n - 1
    } as isize;
    (0..segments).map(move |s| {
        let (p0, p1, p2, p3) = (at(s - 1), at(s), at(s + 1), at(s + 2));
        let k = ((p2 - p1).length() / step).ceil().clamp(1.0, MAX_SEGMENT_SAMPLES) as usize;
        (1..=k).map(move |j| {
            let t = j as f32 / k as f32;
            let (t2, t3) = (t * t, t * t * t);
            0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
        })
    })
}

/// A closed polygon's coverage: 1 inside, 0 outside, a one-unit ramp across the edge.
#[derive(Clone, Debug)]
pub struct Lasso {
    poly: Vec<Vec2>,
    lo: Vec2,
    hi: Vec2,
}

impl Lasso {
    /// Fewer than three finite points cover nothing.
    pub fn new(poly: Vec<Vec2>) -> Lasso {
        let poly: Vec<Vec2> = if poly.len() >= 3 && poly.iter().all(|p| p.is_finite()) { poly } else { Vec::new() };
        let lo = poly.iter().fold(Vec2::splat(f32::MAX), |a, p| a.min(*p)) - 1.0;
        let hi = poly.iter().fold(Vec2::splat(f32::MIN), |a, p| a.max(*p)) + 1.0;
        Lasso { poly, lo, hi }
    }

    pub fn coverage(&self, p: Vec2) -> f32 {
        if !(p.x >= self.lo.x && p.y >= self.lo.y && p.x <= self.hi.x && p.y <= self.hi.y) {
            return 0.0;
        }
        let mut inside = false;
        let mut d2 = f32::MAX;
        for (a, b) in self.poly.iter().zip(self.poly.iter().cycle().skip(1)) {
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
            let e = *b - *a;
            let t = ((p - *a).dot(e) / e.length_squared().max(1e-12)).clamp(0.0, 1.0);
            d2 = d2.min((*a + e * t).distance_squared(p));
        }
        let d = d2.sqrt();
        (if inside { 0.5 + d } else { 0.5 - d }).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_passes_through_the_points() {
        let pts = [Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0), Vec2::new(20.0, 0.0)];
        let c = spline(&pts, false, 1.0);
        for p in pts {
            assert!(c.iter().any(|q| q.distance(p) < 1e-4), "{p}");
        }
        assert_eq!(c.last().copied(), Some(pts[2]));
        let closed = spline(&pts, true, 1.0);
        assert!(closed.last().is_some_and(|q| q.distance(pts[0]) < 1e-4));
        assert!(closed.len() > c.len());
        assert_eq!(spline(&pts[..1], false, f32::NAN), vec![pts[0]]);
    }

    #[test]
    fn lasso_covers_the_inside_with_a_soft_edge() {
        let sq = Lasso::new(vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0)]);
        assert_eq!(sq.coverage(Vec2::new(5.0, 5.0)), 1.0);
        assert_eq!(sq.coverage(Vec2::new(12.0, 5.0)), 0.0);
        assert!((sq.coverage(Vec2::new(10.0, 5.0)) - 0.5).abs() < 1e-4);
        assert_eq!(Lasso::new(vec![Vec2::ZERO, Vec2::ONE]).coverage(Vec2::ZERO), 0.0);
    }
}
