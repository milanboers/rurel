use rurel::mdp::*;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::FixedIterations;
use rurel::*;

// MyState and MyAction implement Serialize and Deserialize so that the data can be saved to JSON or any other file type.

#[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
struct MyState {
    x: i32,
    y: i32,
}
#[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
struct MyAction {
    dx: i32,
    dy: i32,
}

impl State for MyState {
    type A = MyAction;

    fn reward(&self) -> f64 {
        // Negative Euclidean distance
        -((((10 - self.x).pow(2) + (10 - self.y).pow(2)) as f64).sqrt())
    }

    fn actions(&self) -> Vec<MyAction> {
        vec![
            MyAction { dx: 0, dy: -1 }, // up
            MyAction { dx: 0, dy: 1 },  // down
            MyAction { dx: -1, dy: 0 }, // left
            MyAction { dx: 1, dy: 0 },  // right
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

fn main() {
    // Creates a trainer and saves its data to a json
    create_and_save();
    // loads the json, trains some more, then saves to the json the new data.
    load_and_save();
}

/// Create a new Agent with no learned values, train it, then save its data to a JSON file.
fn create_and_save() {
    // Create the trainer object
    let mut trainer = AgentTrainer::new();
    // Starting state of the Agent
    let mut agent = MyAgent {
        state: MyState { x: 0, y: 0 },
    };

    // Training configuration
    trainer.train(
        &mut agent,
        &QLearning::new(0.2, 0.01, 2.),
        &mut FixedIterations::new(100000),
        &RandomExploration::new(),
    );

    // tell the trainer to export its learned values to the specified JSON file.
    trainer.export_learned_values_to_json("grid.json");
}

/// Load existing data into an Agent to continue training, train it some more, then save its data to a JSON file.
fn load_and_save() {
    // Create the trainer object.
    let mut trainer = AgentTrainer::new();
    // import data from the previous training session into the agent.
    trainer.import_state_from_json("grid.json");
    // Set the agents starting state.
    let mut agent = MyAgent {
        state: MyState { x: 0, y: 0 },
    };

    // Training configuration
    trainer.train(
        &mut agent,
        &QLearning::new(0.2, 0.01, 2.),
        &mut FixedIterations::new(0),
        &RandomExploration::new(),
    );

    // Export learned values to specified JSON file.
    trainer.export_learned_values_to_json("grid.json");
}
