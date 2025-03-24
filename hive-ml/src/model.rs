use crate::hypers::{INPUT_ENCODED_DIMS, OUTPUT_LENGTH};
use tch::{nn, Tensor};

pub struct HiveModel {
    pub device: tch::Device,
    shared_layers: tch::nn::Sequential,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
}

impl HiveModel {
    pub fn new(p: &nn::Path) -> Self {
        let stride = |s| nn::ConvConfig {
            stride: s,
            ..Default::default()
        };

        let shared_layers = nn::seq()
            .add(nn::conv2d(
                p / "c1",
                INPUT_ENCODED_DIMS as i64,
                32,
                3,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, stride(2)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 2304, 512, Default::default()))
            .add_fn(|xs| xs.relu());

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", 512, 1, Default::default()))
            .add_fn(|xs| 2 * xs.sigmoid() - 1.0);

        let policy_layer = nn::seq().add(nn::linear(
            p / "al",
            512,
            OUTPUT_LENGTH as i64,
            Default::default(),
        ));

        Self {
            shared_layers,
            policy_layer,
            value_layer,
            device: p.device(),
        }
    }

    pub fn value_policy(&self, game_state: &Tensor) -> (Tensor, Tensor) {
        let t = self.shared_layers(game_state);
        (t.apply(&self.value_layer), t.apply(&self.policy_layer))
    }

    pub fn policy(&self, game_state: &Tensor) -> Tensor {
        let t = self.shared_layers(game_state);
        t.apply(&self.policy_layer)
    }

    fn shared_layers(&self, game_state: &Tensor) -> Tensor {
        game_state.apply(&self.shared_layers)
    }
}
