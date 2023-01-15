/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module for the fixed iterations strategy.

use crate::mdp::State;
use crate::strategy::terminate::TerminationStrategy;

/// The termination strategy that ends if it's at a terminal state (no actions)
pub struct SinkStates {}

impl<S: State> TerminationStrategy<S> for SinkStates {
    fn should_stop(&mut self, state: &S) -> bool {
        state.actions().is_empty()
    }
}
