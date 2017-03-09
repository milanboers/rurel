/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate rand;

use std::hash::Hash;

pub trait State: Eq + Hash + Clone {
    type A: Eq + Hash + Clone;

    fn reward(&self) -> f64;
    fn actions(&self) -> Vec<Self::A>;
    fn random_action(&self) -> Self::A {
        let actions = self.actions();
        let a_t = rand::random::<usize>() % actions.len();
        actions[a_t].clone()
    }
}

pub trait Agent<S: State> {
    fn current_state(&self) -> &S;
    fn take_action(&mut self, &S::A) -> ();
    fn take_random_action(&mut self) -> S::A {
        let action = self.current_state().random_action();
        self.take_action(&action);
        action
    }
}
