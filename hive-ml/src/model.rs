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
        // let stride = |s| nn::ConvConfig {
        //     stride: s,
        //     ..Default::default()
        // };

        let i = INPUT_ENCODED_DIMS as i64;
        let shared_layers = nn::seq()
            // First two layers are 16 channel convolution
            .add(nn::conv2d(p / "c1", i, 16, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 16, 16, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            // Max pool and then 32 channel convolutions
            .add_fn(|xs| xs.max_pool2d_default(2))
            .add(nn::conv2d(p / "c3", 16, 32, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c4", 32, 32, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            // Max pool and then flatten.
            // .add_fn(|xs| xs.max_pool2d_default(2))
            .add_fn(|xs| xs.flat_view())
            // .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 3200, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 256, 256, Default::default()))
            .add_fn(|xs| xs.relu());

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", 256, 1, Default::default()))
            .add_fn(|xs| xs.sigmoid() - 0.5);

        let policy_layer = nn::seq().add(nn::linear(
            p / "al",
            256,
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
