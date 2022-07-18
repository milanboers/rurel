/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module for the fixed iterations strategy.

use crate::mdp::State;
use crate::strategy::terminate::TerminationStrategy;

/// The termination strategy that ends after a certain number of iterations, regardless of the
/// `State`.
pub struct FixedIterations {
    i: u32,
    iters: u32,
}

impl FixedIterations {
    /// Constructs a new termination strategy which ends when `iters` value updates have occurred.
    pub fn new(iters: u32) -> FixedIterations {
        FixedIterations { i: 0, iters }
    }
}

impl<S: State> TerminationStrategy<S> for FixedIterations {
    fn should_stop(&mut self, _: &S) -> bool {
        self.i += 1;
        self.i > self.iters
    }
}
