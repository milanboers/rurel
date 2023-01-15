/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module containing termination strategies.

pub use self::fixed_iterations::FixedIterations;
pub use self::sink_states::SinkStates;
use crate::mdp::State;

pub mod fixed_iterations;
pub mod sink_states;

/// A termination strategy decides when to end training.
pub trait TerminationStrategy<S: State> {
    /// If `should_stop` returns `true`, training will end.
    fn should_stop(&mut self, state: &S) -> bool;
}
