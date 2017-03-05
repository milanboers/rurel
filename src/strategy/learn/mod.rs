/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub mod q;

use std::collections::HashMap;
use mdp::State;

pub trait LearningStrategy<S: State> {
    fn value(&self, &Option<&HashMap<S::Action, f64>>, &Option<&f64>, f64) -> f64;
}
