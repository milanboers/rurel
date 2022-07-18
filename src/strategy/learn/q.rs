/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module for the Q Learning strategy.

use std::collections::HashMap;

use crate::mdp::State;
use crate::strategy::learn::LearningStrategy;

/// The Q Learning strategy
pub struct QLearning {
    alpha: f64,
    gamma: f64,
    initial_value: f64,
}

impl QLearning {
    /// Constructs the Q Learning strategy, with learning rate `alpha`, discount factor `gamma` and
    /// the initial value for Q `initial_value`.
    pub fn new(alpha: f64, gamma: f64, initial_value: f64) -> QLearning {
        QLearning {
            alpha,
            gamma,
            initial_value,
        }
    }
}

impl<S: State> LearningStrategy<S> for QLearning {
    fn value(
        &self,
        new_action_values: &Option<&HashMap<S::A, f64>>,
        old_value: &Option<&f64>,
        reward_after_action: f64,
    ) -> f64 {
        let max_next = new_action_values
            .and_then(|m| m.values().max_by(|a, b| a.partial_cmp(b).unwrap()))
            .unwrap_or(&self.initial_value);
        old_value.map_or(self.initial_value, |x| {
            x + self.alpha * (reward_after_action + self.gamma * max_next - x)
        })
    }
}
