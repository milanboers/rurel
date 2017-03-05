/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub mod mdp;
pub mod strategy;

use std::collections::HashMap;
use mdp::{Agent, State};
use strategy::explore::ExplorationStrategy;
use strategy::learn::LearningStrategy;

pub struct AgentTrainer<S>
    where S: State
{
    q: HashMap<S, HashMap<S::Action, f64>>,
}

impl<S> AgentTrainer<S>
    where S: State
{
    pub fn new() -> AgentTrainer<S> {
        AgentTrainer { q: HashMap::new() }
    }
    // XXX: make associated const with empty map and remove Option?
    pub fn expected_values(&self, state: &S) -> Option<&HashMap<S::Action, f64>> {
        self.q.get(state)
    }
    pub fn expected_value(&self, state: &S, action: &S::Action) -> Option<f64> {
        self.q
            .get(state)
            .and_then(|m| {
                m.get(action)
                    .and_then(|&v| Some(v))
            })
    }
    pub fn train(&mut self,
                 exploration_strategy: &ExplorationStrategy<S>,
                 learning_strategy: &LearningStrategy<S>,
                 agent: &mut Agent<S>,
                 iters: u32)
                 -> () {
        for _ in 1..iters {
            let s_t = agent.current_state().clone();
            let action = exploration_strategy.take_action(agent);

            // current action value
            let s_t_next = agent.current_state();
            let r_t_next = s_t_next.reward();

            let v = {
                let old_value = self.q.get(&s_t).and_then(|m| m.get(&action));
                learning_strategy.value(&self.q.get(s_t_next), &old_value, r_t_next)
            };

            self.q.entry(s_t).or_insert_with(|| HashMap::new()).insert(action, v);
        }
    }
}
