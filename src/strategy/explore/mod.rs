/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Module containing exploration strategies.

pub use self::random::RandomExploration;
use crate::mdp::{Agent, State};

pub mod random;

/// Trait for exploration strategies. An exploration strategy decides, based on an `Agent`, which
/// action to take next.
pub trait ExplorationStrategy<S: State> {
    /// Selects the next action to take for this `Agent`.
    fn pick_action(&self, _: &mut dyn Agent<S>) -> S::A;
}
