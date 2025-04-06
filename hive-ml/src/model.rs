use crate::hypers::{INPUT_ENCODED_DIMS, OUTPUT_LENGTH};
use tch::{nn, Tensor};

pub struct HiveModel {
    pub device: tch::Device,
    train_mode: bool,
    shared_layers: tch::nn::SequentialT,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
}

impl HiveModel {
    pub fn new(p: &nn::Path) -> Self {
        let stride = |s| nn::ConvConfig {
            stride: s,
            ..Default::default()
        };

        let i = INPUT_ENCODED_DIMS as i64;
        let shared_layers = nn::seq_t()
            // First two layers are 16 channel convolution
            .add(nn::conv2d(p / "c1", i, 16, 5, Default::default()))
            .add(nn::batch_norm2d(p / "b1", 16, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d_default(2))
            .add(nn::conv2d(p / "c2", 16, 32, 5, stride(1)))
            .add(nn::batch_norm2d(p / "b2", 32, Default::default()))
            .add_fn(|xs| xs.relu())
            // Max pool and then 32 channel convolutions
            // .add_fn(|xs| xs.max_pool2d_default(2))
            .add(nn::conv2d(p / "c3", 32, 64, 5, stride(1)))
            .add(nn::batch_norm2d(p / "b3", 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c4", 64, 1024, 3, stride(1)))
            .add(nn::batch_norm2d(p / "b4", 1024, Default::default()))
            .add_fn(|xs| xs.relu())
            // Max pool and then flatten.
            // .add_fn(|xs| xs.max_pool2d_default(2))
            .add_fn(|xs| xs.flat_view())
            // .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 1024, 1024, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 1024, 1024, Default::default()))
            .add_fn(|xs| xs.relu());

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", 1024, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "c2", 512, 1, Default::default()))
            .add_fn(|xs| xs.sigmoid() - 0.5);

        let policy_layer = nn::seq()
            .add(nn::linear(p / "a1", 1024, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                p / "a2",
                512,
                OUTPUT_LENGTH as i64,
                Default::default(),
            ));

        Self {
            shared_layers,
            policy_layer,
            value_layer,
            train_mode: false,
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

    pub fn set_train_mode(&mut self, train_mode: bool) {
        self.train_mode = train_mode;
    }

    fn shared_layers(&self, game_state: &Tensor) -> Tensor {
        game_state.apply_t(&self.shared_layers, self.train_mode)
    }
}
