//! The five handles of the levels editor and how each one reads and moves a [`Levels`]

use texturecraft_engine::core::Levels;

use crate::i18n::tr;

/// Narrowest gap the handles and typed values leave between the low and high handle of a pair
const MIN_HANDLE_GAP: f32 = 0.01;

/// Gamma range of the handle and typed values: the engine's own, so the handle nearly reaches either input handle
const GAMMA_RANGE: std::ops::RangeInclusive<f32> = 0.01..=10.0;

/// One draggable value of a levels effect
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LevelsHandle {
    InputLow,
    Gamma,
    InputHigh,
    OutputLow,
    OutputHigh,
}

impl LevelsHandle {
    /// The handles above the histogram, left to right at the default levels
    pub const INPUT: [LevelsHandle; 3] = [LevelsHandle::InputLow, LevelsHandle::Gamma, LevelsHandle::InputHigh];

    /// The localized name shared by the handle and its number field
    pub fn label(self) -> &'static str {
        match self {
            LevelsHandle::InputLow => tr("Input low"),
            LevelsHandle::Gamma => tr("Gamma"),
            LevelsHandle::InputHigh => tr("Input high"),
            LevelsHandle::OutputLow => tr("Output low"),
            LevelsHandle::OutputHigh => tr("Output high"),
        }
    }

    /// The stored value, before display rounding
    pub fn value(self, levels: &Levels) -> f32 {
        match self {
            LevelsHandle::InputLow => levels.in_low,
            LevelsHandle::Gamma => levels.gamma,
            LevelsHandle::InputHigh => levels.in_high,
            LevelsHandle::OutputLow => levels.out_low,
            LevelsHandle::OutputHigh => levels.out_high,
        }
    }

    /// Where the handle sits along its track (0 left, 1 right); gamma 1 is halfway between the input handles
    pub fn position(self, levels: &Levels) -> f32 {
        match self {
            LevelsHandle::Gamma => levels.in_low + input_span(levels) * 0.5f32.powf(levels.gamma),
            _ => self.value(levels),
        }
        .clamp(0.0, 1.0)
    }

    /// Moves the handle to `position` along its track (see [`Self::position`])
    pub fn set_position(self, levels: &mut Levels, position: f32) {
        let value = match self {
            LevelsHandle::Gamma => {
                let span = input_span(levels).max(f32::EPSILON);
                let fraction = ((position - levels.in_low) / span).clamp(f32::EPSILON, 1.0 - f32::EPSILON);
                fraction.ln() / 0.5f32.ln()
            }
            _ => position,
        };
        self.set_value(levels, value);
    }

    /// Allowed values enforce ordered inputs and preserve inverted outputs
    pub fn value_range(self, levels: &Levels) -> std::ops::RangeInclusive<f32> {
        let (neighbour, is_lower) = match self {
            LevelsHandle::InputLow => (levels.in_high, true),
            LevelsHandle::InputHigh => (levels.in_low, false),
            LevelsHandle::OutputLow => (levels.out_high, levels.out_low <= levels.out_high),
            LevelsHandle::OutputHigh => (levels.out_low, levels.out_high < levels.out_low),
            LevelsHandle::Gamma => return GAMMA_RANGE,
        };

        // Invalid project values must not become NaN clamp bounds
        if !neighbour.is_finite() {
            return 0.0..=1.0;
        }

        // Saved inputs can cross; clamp each bound so editing them cannot panic
        if is_lower { 0.0..=(neighbour - MIN_HANDLE_GAP).clamp(0.0, 1.0) } else { (neighbour + MIN_HANDLE_GAP).clamp(0.0, 1.0)..=1.0 }
    }

    /// Sets a finite value without crossing the neighbouring endpoint
    pub fn set_value(self, levels: &mut Levels, value: f32) {
        if !value.is_finite() {
            return;
        }

        let range = self.value_range(levels);
        let value = value.clamp(*range.start(), *range.end());
        match self {
            LevelsHandle::InputLow => levels.in_low = value,
            LevelsHandle::Gamma => levels.gamma = value,
            LevelsHandle::InputHigh => levels.in_high = value,
            LevelsHandle::OutputLow => levels.out_low = value,
            LevelsHandle::OutputHigh => levels.out_high = value,
        }
    }
}

fn input_span(levels: &Levels) -> f32 {
    (levels.in_high - levels.in_low).max(0.0)
}
