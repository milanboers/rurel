/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Rurel is a flexible, reusable reinforcement learning (Q learning) implementation in Rust.
//!
//! Implement the [Agent](mdp/trait.Agent.html) and [State](mdp/trait.State.html) traits for your
//! process, then create an [AgentTrainer](struct.AgentTrainer.html) and train it for your process.
//!
//! # Basic Example
//!
//! The following example defines the `State` as a position on a 21x21 2D matrix. The `Action`s
//! that can be taken are: go up, go down, go left and go right. Positions closer to (10, 10) are
//! assigned a higher reward.
//!
//! After training, the AgentTrainer will have assigned higher values to actions which move closer
//! to (10, 10).
//!
//! ```
//! use rurel::mdp::{State, Agent};
//!
//! #[derive(PartialEq, Eq, Hash, Clone)]
//! struct MyState { x: i32, y: i32 }
//! #[derive(PartialEq, Eq, Hash, Clone)]
//! struct MyAction { dx: i32, dy: i32 }
//!
//! impl State for MyState {
//!     type A = MyAction;
//!     fn reward(&self) -> f64 {
//!         // Negative Euclidean distance
//!         -((((10 - self.x).pow(2) + (10 - self.y).pow(2)) as f64).sqrt())
//!     }
//!     fn actions(&self) -> Vec<MyAction> {
//!         vec![MyAction { dx: 0, dy: -1 },    // up
//!              MyAction { dx: 0, dy: 1 }, // down
//!              MyAction { dx: -1, dy: 0 },    // left
//!              MyAction { dx: 1, dy: 0 }, // right
//!         ]
//!     }
//! }
//!
//! struct MyAgent { state: MyState }
//! impl Agent<MyState> for MyAgent {
//!     fn current_state(&self) -> &MyState {
//!         &self.state
//!     }
//!     fn take_action(&mut self, action: &MyAction) -> () {
//!         match action {
//!             &MyAction { dx, dy } => {
//!                 self.state = MyState {
//!                     x: (((self.state.x + dx) % 21) + 21) % 21, // (x+dx) mod 21
//!                     y: (((self.state.y + dy) % 21) + 21) % 21, // (y+dy) mod 21
//!                 }
//!             }
//!         }
//!     }
//! }
//!
//! use rurel::AgentTrainer;
//! use rurel::strategy::learn::QLearning;
//! use rurel::strategy::explore::RandomExploration;
//! use rurel::strategy::terminate::FixedIterations;
//!
//! let mut trainer = AgentTrainer::new();
//! let mut agent = MyAgent { state: MyState { x: 0, y: 0 }};
//! trainer.train(&mut agent,
//!               &QLearning::new(0.2, 0.01, 2.),
//!               &mut FixedIterations::new(100000),
//!               &RandomExploration::new());
//!
//! // Test to see if it worked
//! let test_state = MyState { x: 10, y: 9 };
//! let go_up = MyAction { dx: 0, dy: -1 };
//! let go_down = MyAction { dx: 0, dy: 1};
//! // Going down is better than going up
//! assert!(trainer.expected_value(&test_state, &go_down)
//!     > trainer.expected_value(&test_state, &go_up));
//! ```

// Run tests in the readme, but don't include readme in the documentation.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod doc_test {}

use std::collections::HashMap;

use mdp::{Agent, State};
use strategy::explore::ExplorationStrategy;
use strategy::learn::LearningStrategy;
use strategy::terminate::TerminationStrategy;

#[cfg(feature = "dqn")]
pub mod dqn;
pub mod mdp;
pub mod strategy;

/// An `AgentTrainer` can be trained for using a certain [Agent](mdp/trait.Agent.html). After
/// training, the `AgentTrainer` contains learned knowledge about the process, and can be queried
/// for this. For example, you can ask the `AgentTrainer` the expected values of all possible
/// actions in a given state.
pub struct AgentTrainer<S>
where
    S: State,
{
    q: HashMap<S, HashMap<S::A, f64>>,
}

impl<S> AgentTrainer<S>
where
    S: State,
{
    pub fn new() -> AgentTrainer<S> {
        AgentTrainer { q: HashMap::new() }
    }

    /// Fetches the learned values for the given state, by `Action`, or `None` if no value was
    /// learned.
    pub fn expected_values(&self, state: &S) -> Option<&HashMap<S::A, f64>> {
        // XXX: make associated const with empty map and remove Option?
        self.q.get(state)
    }

    /// Fetches the learned value for the given `Action` in the given `State`, or `None` if no
    /// value was learned.
    pub fn expected_value(&self, state: &S, action: &S::A) -> Option<f64> {
        self.q.get(state).and_then(|m| m.get(action).copied())
    }

    /// Returns a clone of the entire learned state to be saved or used elsewhere.
    pub fn export_learned_values(&self) -> HashMap<S, HashMap<S::A, f64>> {
        self.q.clone()
    }

    // Returns a reference to the learned state.
    pub fn learned_values(&self) -> &HashMap<S, HashMap<S::A, f64>> {
        &self.q
    }

    /// Imports a state, completely replacing any learned progress
    pub fn import_state(&mut self, q: HashMap<S, HashMap<S::A, f64>>) {
        self.q = q;
    }

    /// Returns the best action for the given `State`, or `None` if no values were learned.
    pub fn best_action(&self, state: &S) -> Option<S::A> {
        self.expected_values(state)
            .and_then(|m| {
                m.iter()
                    .max_by(|&(_, v1), &(_, v2)| v1.partial_cmp(v2).unwrap())
            })
            .map(|t| t.0.clone())
    }

    /// Trains this [AgentTrainer] using the given [ExplorationStrategy], [LearningStrategy] and
    /// [Agent] until the [TerminationStrategy] decides to stop.
    pub fn train(
        &mut self,
        agent: &mut dyn Agent<S>,
        learning_strategy: &dyn LearningStrategy<S>,
        termination_strategy: &mut dyn TerminationStrategy<S>,
        exploration_strategy: &dyn ExplorationStrategy<S>,
    ) {
        loop {
            let s_t = agent.current_state().clone();
            let action = exploration_strategy.pick_action(agent);

            // current action value
            let s_t_next = agent.current_state();
            let r_t_next = s_t_next.reward();

            let v = {
                let old_value = self.q.get(&s_t).and_then(|m| m.get(&action));
                learning_strategy.value(&self.q.get(s_t_next), &old_value, r_t_next)
            };

            self.q.entry(s_t).or_default().insert(action, v);

            if termination_strategy.should_stop(s_t_next) {
                break;
            }
        }
    }
}

impl<S: State> Default for AgentTrainer<S> {
    fn default() -> Self {
        Self::new()
    }
}
