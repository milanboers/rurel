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
//! use rurel::*;
//!
//! #[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
//! struct MyState { x: i32, y: i32 }
//! #[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
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
use std::fs;

// making serde available from this crate
// serde is necessary to parse the created data into
// a json file or some other serialized format
pub extern crate serde;
// making de and se public so it can be accessed using rurel::*;
pub use serde::{Deserialize, Serialize};
use serde::ser::SerializeSeq;

use mdp::{Agent, State};
use strategy::explore::ExplorationStrategy;
use strategy::learn::LearningStrategy;
use strategy::terminate::TerminationStrategy;

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

    /// writes the data to the indicated JSON file
    pub fn export_learned_values_to_json(&self, filepath: &str) {
        use std::io::Write;
        // Create / Overwrite file
        let mut file = fs::File::create(filepath)
            .expect("Failed to Create File!");
        // write the serialized data to JSON then to the file
        file.write_all(serde_json::to_string(&self).unwrap().as_bytes())
            .expect("Failed to write json data to file");
    }

    /// Imports a state, completely replacing any learned progress
    pub fn import_state(&mut self, q: HashMap<S, HashMap<S::A, f64>>) {
        self.q = q;
    }

    /// Imports a state from a JSON file, completely replacing any learned progress.  
    pub fn import_state_from_json(&mut self, filepath: &str) {
        // Get string from file
        let contents = std::fs::read_to_string("data/grid.json")
            .expect("Failed to read file contents!");
        // initialize data container
        let mut data: HashMap<S, HashMap<S::A, f64>> = HashMap::new();

        // if the contents are empty (no data exists), just begin with an empty hashmap
        // Else, pull everything out of the file and put it into data.
        if contents != "" {
            // deserialize from vector data
            let vec_data: Vec<(S, Vec<(S::A, f64)>)> = serde_json::from_str(&contents)
                .expect("Failed to parse JSON File!");
            // temporary hashmap for holding inner data
            let mut inner: HashMap<S::A, f64> = HashMap::new();
            // iterate through the vector
            for (state, actions) in vec_data.iter() {
                // clear inner data so no duplicates
                inner.clear();
                for (a, r) in actions.iter() {
                    // put data in inner
                    inner.insert(a.clone(), r.clone());
                }
                // put state and inner in data
                data.insert(state.clone(), inner.clone());
            }
        }

        self.q = data;
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

            self.q
                .entry(s_t)
                .or_insert_with(HashMap::new)
                .insert(action, v);

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

/// Added to serialize the data within AgentTrainer for saving and loading.
impl<S> Serialize for AgentTrainer<S>
where
    S: State,
{
    fn serialize<SER>(&self, serializer: SER) -> Result<SER::Ok, SER::Error>
    where
        SER: serde::Serializer,
    {
        // Container we want to convert self.q into
        let mut vec_data: Vec<(S, Vec<(S::A, f64)>)> = Vec::with_capacity(self.q.len());
        // temporary vector for holding inner data
        let mut inner: Vec<(S::A, f64)> = Vec::new();
        // Iterate over self.q
        for (state, actions) in self.q.iter() {
            // clear inner so there are no duplicates
            inner.clear();
            for (a, r) in actions.iter() {
                // fill inner with necessary contents
                inner.push((a.clone(), *r));
            }
            // fill vector data
            vec_data.push((state.clone(), inner.clone()));
        }

        // initialize serializer as a sequence, which we can do now since it is a Vec
        let mut seq = serializer.serialize_seq(Some(self.q.len()))?;
        // iterate over vec data and serialize each element into the serializer
        for e in vec_data.iter() {
            seq.serialize_element(&e)?;
        }
        // end serialization
        seq.end()
    }
}
