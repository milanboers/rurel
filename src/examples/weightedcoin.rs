/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use rurel::mdp::{Agent, State};
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::SinkStates;
use rurel::AgentTrainer;

const TARGET: i32 = 100;
const WEIGHT: u8 = 100; //portion of 255

#[derive(PartialEq, Eq, Hash, Clone)]
struct CoinState {
    balance: i32,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct CoinAction {
    bet: i32,
}

impl State for CoinState {
    type A = CoinAction;

    fn reward(&self) -> f64 {
        if self.balance >= TARGET {
            1.0
        } else {
            0.0
        }
    }

    fn actions(&self) -> Vec<CoinAction> {
        let bet_range = {
            if self.balance < TARGET / 2 {
                1..self.balance + 1
            } else {
                1..(TARGET - self.balance) + 1
            }
        };
        bet_range.map(|bet| CoinAction { bet }).collect()
    }
}

struct CoinAgent {
    state: CoinState,
}

impl Agent<CoinState> for CoinAgent {
    fn current_state(&self) -> &CoinState {
        &self.state
    }
    fn take_action(&mut self, action: &CoinAction) {
        //Update the state to:
        self.state = CoinState {
            balance: if rand::random::<u8>() <= WEIGHT {
                self.state.balance + action.bet
            }
            //If the coin is heads, balance + bet
            else {
                self.state.balance - action.bet
            }, //If the coin is tails, balance - bet
        }
    }
}

fn main() {
    const TRIALS: i32 = 100000;
    let mut trainer = AgentTrainer::new();
    for trial in 0..TRIALS {
        let mut agent = CoinAgent {
            state: CoinState {
                balance: 1 + trial % 98,
            },
        };
        trainer.train(
            &mut agent,
            &QLearning::new(0.2, 1.0, 0.0),
            &mut SinkStates {},
            &RandomExploration::new(),
        );
    }

    println!("Balance\tBet\tQ-value");
    for balance in 1..TARGET {
        let state = CoinState { balance };
        let action = trainer.best_action(&state).unwrap();
        println!(
            "{}\t{}\t{}",
            balance,
            action.bet,
            trainer.expected_value(&state, &action).unwrap(),
        );
    }
}
