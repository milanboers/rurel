/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub mod random;

use mdp::{State, Agent};

pub trait ExplorationStrategy<S: State> {
    fn take_action(&self, &mut Agent<S>) -> S::Action;
}
