/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use mdp::{State, Agent};
use strategy::explore::ExplorationStrategy;

/// The random exploration strategy. This strategy always takes a random action, as defined for the
/// Agent by
/// [Agent::take_random_action()](../../../mdp/trait.Agent.html#method.take_random_action)
pub struct RandomExploration;

impl RandomExploration {
    pub fn new() -> RandomExploration {
        RandomExploration
    }
}

impl<S: State> ExplorationStrategy<S> for RandomExploration {
    fn pick_action(&self, agent: &mut Agent<S>) -> S::A {
        agent.pick_random_action()
    }
}
