use crate::hypers;
use tch::{nn, Tensor};

pub struct HiveModel {
    pub device: tch::Device,
    train_mode: bool,
    conv_i: tch::nn::Conv2D,
    bn_i: tch::nn::BatchNorm,
    residual_blocks: Vec<(tch::nn::Conv2D, tch::nn::BatchNorm)>,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,

    in_value_layer: tch::nn::Sequential,
    in_distill_layer: tch::nn::Sequential,
    in_random_layer: tch::nn::Sequential,
}

impl HiveModel {
    pub fn new(p: &nn::Path) -> Self {
        const NUM_BLOCKS: usize = 9;

        // let stride = |s| nn::ConvConfig {
        //     stride: s,
        //     ..Default::default()
        //};

        let input_encoded_dims = hypers::INPUT_ENCODED_DIMS as i64;
        let channels = 64;

        let conv_i = tch::nn::conv2d(
            p / "l0" / "conv",
            input_encoded_dims,
            channels,
            3,
            tch::nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );

        let bn_i = tch::nn::batch_norm2d(p / "l0" / "bn", channels, Default::default());

        let residual_blocks = (1..=NUM_BLOCKS)
            .map(|idx| {
                let layer = format!("l{}", idx);

                let conv_layer = tch::nn::conv2d(
                    p / layer.as_str() / "conv",
                    channels,
                    channels,
                    3,
                    tch::nn::ConvConfig {
                        padding: 1,
                        ..Default::default()
                    },
                );

                let bn_layer =
                    tch::nn::batch_norm2d(p / layer.as_str() / "bn", channels, Default::default());

                (conv_layer, bn_layer)
            })
            .collect::<Vec<_>>();

        let dims_after_conv = 2304;
        let policy_layer = nn::seq().add(nn::linear(
            p / "a" / "l1",
            dims_after_conv,
            hypers::OUTPUT_LENGTH as i64,
            Default::default(),
        ));

        let value_layer = nn::seq()
            .add(nn::linear(
                p / "c" / "l1",
                dims_after_conv,
                1,
                Default::default(),
            ))
            .add_fn(|xs| xs.tanh());

        // Layers for calculating the distillation + intrinsic rewards.
        let in_value_layer = nn::seq().add(nn::linear(
            p / "in" / "l1",
            dims_after_conv,
            1,
            Default::default(),
        ));

        let in_random_layer = nn::seq()
            .add(nn::conv2d(
                p / "in" / "r1",
                input_encoded_dims,
                1,
                1,
                Default::default(),
            ))
            .add_fn(|xs| xs.flat_view())
            .add(nn::linear(p / "in" / "l2", 26 * 26, 1, Default::default()));

        let in_distill_layer = nn::seq()
            .add(nn::conv2d(
                p / "in" / "r1",
                input_encoded_dims,
                1,
                1,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.flat_view())
            .add(nn::linear(p / "in" / "l2", 26 * 26, 1, Default::default()));

        Self {
            policy_layer,
            value_layer,
            in_value_layer,
            in_distill_layer,
            in_random_layer,
            conv_i,
            bn_i,
            residual_blocks,
            train_mode: false,
            device: p.device(),
        }
    }

    pub fn value_policy(&self, game_state: &Tensor) -> (Tensor, Tensor) {
        let t = self.shared_layers(game_state);
        (
            t.detach().apply(&self.value_layer),
            t.apply(&self.policy_layer),
        )
    }

    pub fn value_intrinsic_value_policy(&self, game_state: &Tensor) -> (Tensor, Tensor, Tensor) {
        let t = self.shared_layers(game_state);
        (
            t.detach().apply(&self.value_layer),
            t.detach().apply(&self.in_value_layer),
            t.apply(&self.policy_layer),
        )
    }

    pub fn value_policy_auxiliary(&self, game_state: &Tensor) -> (Tensor, Tensor) {
        let t = self.shared_layers(game_state);
        (t.apply(&self.value_layer), t.apply(&self.policy_layer))
    }

    pub fn policy(&self, game_state: &Tensor) -> Tensor {
        let t = self.shared_layers(game_state);
        t.apply(&self.policy_layer)
    }

    pub fn novelty(&self, game_state: &Tensor) -> Tensor {
        let distilled = game_state.apply(&self.in_distill_layer);
        let random = game_state.apply(&self.in_random_layer).detach();
        (distilled - random).square()
    }

    pub fn set_train_mode(&mut self, train_mode: bool) {
        self.train_mode = train_mode;
    }

    fn shared_layers(&self, game_state: &Tensor) -> Tensor {
        let mut output = game_state.apply(&self.conv_i);
        output = output.apply_t(&self.bn_i, self.train_mode);
        output = output.max_pool2d_default(2);

        for (c, bn) in &self.residual_blocks {
            output = &output + output.apply(c).tanh();
            output = output.apply_t(bn, self.train_mode);
        }

        output = output.max_pool2d_default(2);
        output.flat_view()
    }
}
