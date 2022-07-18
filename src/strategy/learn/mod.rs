/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module containing learning (value updating) strategies.

use std::collections::HashMap;

pub use self::q::QLearning;
use crate::mdp::State;

pub mod q;

/// A learning strategy can calculate a learned value for the action which was taken from the
/// values for the actions in the new state (`new_action_values`), the current value
/// (`current_value`), and the reward that was received after taking the action.
pub trait LearningStrategy<S: State> {
    /// Calculates a learned value for the action which was taken from the
    /// values for the actions in the new state (`new_action_values`), the current value
    /// (`current_value`), and the reward that was received after taking the action.
    fn value(
        &self,
        new_action_values: &Option<&HashMap<S::A, f64>>,
        current_value: &Option<&f64>,
        received_reward: f64,
    ) -> f64;
}
