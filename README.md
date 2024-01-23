# Rurel

[![crates.io](https://img.shields.io/crates/v/rurel.svg)](https://crates.io/crates/rurel)

Rurel is a flexible, reusable reinforcement learning (Q learning) implementation in Rust.

* [Release documentation](https://docs.rs/rurel)

In Cargo.toml:
```toml
rurel = "0.5.1"
```


An example is included. This teaches an agent on a 21x21 grid how to arrive at 10,10, using actions (go left, go up, go right, go down):
```console
cargo run --example eucdist
```

## Getting started
There are two main traits you need to implement: `rurel::mdp::State` and `rurel::mdp::Agent`.

A `State` is something which defines a `Vec` of actions that can be taken from this state, and has a certain reward. A `State` needs to define the corresponding action type `A`.

An `Agent` is something which has a current state, and given an action, can take the action and evaluate the next state.

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
	fn reward(&self) -> f64 {
		// Negative Euclidean distance
		-((((10 - self.x).pow(2) + (10 - self.y).pow(2)) as f64).sqrt())
	}
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

struct MyAgent { state: MyState }
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
}
```

That's all. Now make a trainer and train the agent with Q learning, with learning rate 0.2, discount factor 0.01 and an initial value of Q of 2.0. We let the trainer run for 100000 iterations, randomly exploring new states.

```rust, ignore
use rurel::AgentTrainer;
use rurel::strategy::learn::QLearning;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::terminate::FixedIterations;

let mut trainer = AgentTrainer::new();
let mut agent = MyAgent { state: MyState { x: 0, y: 0 }};
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
* Run `cargo fmt --all` to format the code.
* Run `cargo clippy --all-targets --all-features -- -Dwarnings` to lint the code.
* Run `cargo test` to test the code.
