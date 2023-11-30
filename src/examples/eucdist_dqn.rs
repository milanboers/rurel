/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "dqn")]
use rurel::dqn::DQNAgentTrainer;
use rurel::mdp::{Agent, State};

/// A simple 2D grid world where the agent can move around.
/// The agent has to reach (10, 10).

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct MyState {
    tx: i32,
    ty: i32,
    x: i32,
    y: i32,
    maxx: i32,
    maxy: i32,
}

// Into float array has to be implemented for the DQN state
impl From<MyState> for [f32; 6] {
    fn from(val: MyState) -> Self {
        [
            val.tx as f32,
            val.ty as f32,
            val.x as f32,
            val.y as f32,
            val.maxx as f32,
            val.maxy as f32,
        ]
    }
}

// From float array has to be implemented for the DQN state
impl From<[f32; 6]> for MyState {
    fn from(v: [f32; 6]) -> Self {
        MyState {
            tx: v[0] as i32,
            ty: v[1] as i32,
            x: v[2] as i32,
            y: v[3] as i32,
            maxx: v[4] as i32,
            maxy: v[5] as i32,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum MyAction {
    Move { dx: i32, dy: i32 },
}

// Into float array has to be implemented for the action,
// so that the DQN can use it.
impl From<MyAction> for [f32; 4] {
    fn from(val: MyAction) -> Self {
        match val {
            MyAction::Move { dx: -1, dy: 0 } => [1.0, 0.0, 0.0, 0.0],
            MyAction::Move { dx: 1, dy: 0 } => [0.0, 1.0, 0.0, 0.0],
            MyAction::Move { dx: 0, dy: -1 } => [0.0, 0.0, 1.0, 0.0],
            MyAction::Move { dx: 0, dy: 1 } => [0.0, 0.0, 0.0, 1.0],
            _ => panic!("Invalid action"),
        }
    }
}

// From float array has to be implemented for the action,
// because output of the DQN is a float array like [0.1, 0.2, 0.1, 0.1]
impl From<[f32; 4]> for MyAction {
    fn from(v: [f32; 4]) -> Self {
        // Find the index of the maximum value
        let max_index = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        match max_index {
            0 => MyAction::Move { dx: -1, dy: 0 },
            1 => MyAction::Move { dx: 1, dy: 0 },
            2 => MyAction::Move { dx: 0, dy: -1 },
            3 => MyAction::Move { dx: 0, dy: 1 },
            _ => panic!("Invalid action index"),
        }
    }
}

impl State for MyState {
    type A = MyAction;

    // The reward is the exponential of the negative distance to the target
    fn reward(&self) -> f64 {
        let (tx, ty) = (self.tx, self.ty);
        let d = (((tx - self.x).pow(2) + (ty - self.y).pow(2)) as f64).sqrt();
        -d
    }

    fn actions(&self) -> Vec<MyAction> {
        vec![
            MyAction::Move { dx: -1, dy: 0 },
            MyAction::Move { dx: 1, dy: 0 },
            MyAction::Move { dx: 0, dy: -1 },
            MyAction::Move { dx: 0, dy: 1 },
        ]
    }
}

struct MyAgent {
    state: MyState,
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }

    fn take_action(&mut self, action: &MyAction) {
        match action {
            &MyAction::Move { dx, dy } => {
                self.state = MyState {
                    x: (((self.state.x + dx) % self.state.maxx) + self.state.maxx)
                        % self.state.maxx,
                    y: (((self.state.y + dy) % self.state.maxy) + self.state.maxy)
                        % self.state.maxy,
                    ..self.state.clone()
                };
            }
        }
    }
}

#[cfg(feature = "dqn")]
fn main() {
    use rurel::strategy::explore::RandomExploration;
    use rurel::strategy::terminate::FixedIterations;
    let (tx, ty) = (10, 10);
    let (maxx, maxy) = (21, 21);
    let initial_state = MyState {
        tx,
        ty,
        x: 0,
        y: 0,
        maxx,
        maxy,
    };

    let mut trainer = DQNAgentTrainer::<MyState, 6, 4, 64>::new(0.9, 1e-3);
    let mut agent = MyAgent {
        state: initial_state.clone(),
    };
    trainer.train(
        &mut agent,
        &mut FixedIterations::new(10_000),
        &RandomExploration::new(),
    );
    for j in 0..maxy {
        for i in 0..maxx {
            let best_action = trainer
                .best_action(&MyState {
                    tx,
                    ty,
                    x: i,
                    y: j,
                    maxx,
                    maxy,
                })
                .unwrap();
            match best_action {
                MyAction::Move { dx: -1, dy: 0 } => print!("<"),
                MyAction::Move { dx: 1, dy: 0 } => print!(">"),
                MyAction::Move { dx: 0, dy: -1 } => print!("^"),
                MyAction::Move { dx: 0, dy: 1 } => print!("v"),
                _ => print!("-"),
            };
        }
        println!();
    }

    /*
    >>>>>vvvvvvvvvv<<<<<<
    >>>>>vvvvvvvvvv<<<<<<
    >>>>>vvvvvvvvv<<<<<<<
    >>>>>>vvvvvvvv<<<<<<<
    >>>>>>vvvvvvvv<<<<<<<
    >>>>>>>vvvvvv<<<<<<<<
    >>>>>>>vvvvvv<<<<<<<<
    >>>>>>>>vvvv<<<<<<<<<
    >>>>>>>>>vv<<<<<<<<<<
    >>>>>>>>>v<<<<<<<<<<<
    >>>>>>>>>^^<<<<<<<<<<
    >>>>>>>>^^^^<<<<<<<<<
    >>>>>>>^^^^^^<<<<<<<<
    >>>>>^^^^^^^^^<<<<<<<
    >>>^^^^^^^^^^^^<<<<<<
    >^^^^^^^^^^^^^^<<<<<<
    ^^^^^^^^^^^^^^^^<<<<<
    ^^^^^^^^^^^^^^^^^<<<<
    ^^^^^^^^^^^^^^^^^^^<<
    ^^^^^^^^^^^^^^^^^^^^<
    ^^^^^^^^^^^^^^^^^^^^^
    */
}

#[cfg(not(feature = "dqn"))]
fn main() {
    panic!("Use the 'dqn' feature to run this example");
}
