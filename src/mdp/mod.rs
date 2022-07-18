/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::hash::Hash;

/// A `State` is something which has a reward, and has a certain set of actions associated with it.
/// The type of the actions must be defined as the associated type `A`.
pub trait State: Eq + Hash + Clone {
    /// Action type associate with this `State`.
    type A: Eq + Hash + Clone;

    /// The reward for when an `Agent` arrives at this `State`.
    fn reward(&self) -> f64;
    /// The set of actions that can be taken from this `State`, to arrive in another `State`.
    fn actions(&self) -> Vec<Self::A>;
    /// Selects a random action that can be taken from this `State`. The default implementation
    /// takes a uniformly distributed random action from the defined set of actions. You may want
    /// to improve the performance by only generating the necessary action.
    fn random_action(&self) -> Self::A {
        let actions = self.actions();
        let a_t = rand::random::<usize>() % actions.len();
        actions[a_t].clone()
    }
}

/// An `Agent` is something which hold a certain state, and is able to take actions from that
/// state. After taking an action, the agent arrives at another state.
pub trait Agent<S: State> {
    /// Returns the current state of this `Agent`.
    fn current_state(&self) -> &S;
    /// Takes the given action, possibly mutating the current `State`.
    fn take_action(&mut self, action: &S::A);
    /// Takes a random action from the set of possible actions from this `State`. The default
    /// implementation uses [State::random_action()](trait.State.html#method.random_action) to
    /// determine the action to be taken.
    fn pick_random_action(&mut self) -> S::A {
        let action = self.current_state().random_action();
        self.take_action(&action);
        action
    }
}
