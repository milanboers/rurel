# Rurel

[![Build Status](https://travis-ci.org/milanboers/rurel.svg?branch=master)](https://travis-ci.org/milanboers/rurel)
[![crates.io](https://img.shields.io/crates/v/rurel.svg)](https://crates.io/crates/rurel)

Rurel is a flexible, reusable reinforcement learning (Q learning) implementation in Rust.

* [Release documentation](https://docs.rs/rurel)

In Cargo.toml:
```toml
rurel = "0.2.1"
```


An example is included. This teaches an agent on a 21x21 grid how to arrive at 10,10, using actions (go left, go up, go right, go down):
```console
cargo run --example eucdist
```

## Getting started
There are two main traits you need to implement: `rurel::mdp::State` and `rurel::mdp::Agent`.

A `State` is something which defines a `Vec` of actions that can be taken from this relevant state. A `State` needs to define the corresponding action type `A`.

An `Agent` is something which has a current state, and given an action, can take the action, evaluate the next state and has a certain reward calculated with relevant (fields from `State`) and irrelevant (fields from `Agent`) state.

### Example

Let's implement the example in `cargo run --example eucdist`. We want to make an agent which is taught how to arrive at 10,10 on a 21x21 grid.

First, let's define a `State`, which should represent a position on a 21x21, and the correspoding Action, which is either up, down, left or right.

```rust
use rurel::mdp::State;

#[derive(PartialEq, Eq, Hash, Clone)]
struct MyState { x: i32, y: i32 }
#[derive(PartialEq, Eq, Hash, Clone)]
struct MyAction { dx: i32, dy: i32 }

impl State for MyState {
	type A = MyAction;
	fn actions(&self) -> Vec<MyAction> {
		vec![MyAction { dx: 0, dy: -1 },	// up
			 MyAction { dx: 0, dy: 1 },	// down
			 MyAction { dx: -1, dy: 0 },	// left
			 MyAction { dx: 1, dy: 0 },	// right
		]
	}
}
```

Then define the agent:

```rust, ignore
use rurel::mdp::Agent;

struct MyAgent { state: MyState, irrelevant_data: bool }
impl Agent<MyState> for MyAgent {
	fn current_state(&self) -> &MyState {
		&self.state
	}
	fn take_action(&mut self, action: &MyAction) -> () {
		match action {
			&MyAction { dx, dy } => {
				self.state = MyState {
					x: (((self.state.x + dx) % 21) + 21) % 21, // (x+dx) mod 21
					y: (((self.state.y + dy) % 21) + 21) % 21, // (y+dy) mod 21
				}
			}
		}
	}
    fn reward(&self) -> f64 {
		// Negative Euclidean distance
        let (tx, ty) = (10, 10);
        let (x, y) = (self.state.x, self.state.y);
        let d = (((tx - x).pow(2) + (ty - y).pow(2)) as f64).sqrt();
        if self.irrelevant_data {
            -d
        } else {
            -d
        }
    }
}
```

That's all. Now make a trainer and train the agent with Q learning, with learning rate 0.2, discount factor 0.01 and an initial value of Q of 2.0. We let the trainer run for 100000 iterations, randomly exploring new states.

```rust, ignore
use rurel::AgentTrainer;
use rurel::strategy::learn::QLearning;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::terminate::FixedIterations;

let mut trainer = AgentTrainer::new();
let mut agent = MyAgent { state: MyState { x: 0, y: 0 }, irrelevant_data: true};
trainer.train(&mut agent,
              &QLearning::new(0.2, 0.01, 2.),
              &mut FixedIterations::new(100000),
              &RandomExploration::new());
```

After this, you can query the learned value (Q) for a certain action in a certain state by:

```rust, ignore
trainer.expected_value(&state, &action) // : Option<f64>
```

## Development
* Run `cargo +nightly fmt` to format the code.
