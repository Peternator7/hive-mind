use crate::hypers::{INPUT_ENCODED_DIMS, OUTPUT_LENGTH};
use tch::{
    nn,
    Tensor,
};

pub struct HiveModel {
    pub device: tch::Device,
    shared_layers: tch::nn::Sequential,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
}

impl HiveModel {
    pub fn new(p: &nn::Path) -> Self {
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

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    packed_proj: nn::Linear,
    output_proj: nn::Linear,
    head_dim: i64,
    nheads: i64,
}

impl MultiHeadSelfAttention {
    pub fn new(p: &nn::Path, embed_dim: i64, total: i64, nheads: i64) -> Self {
        let packed_proj = nn::linear(p, embed_dim, total * 3, Default::default());
        let output_proj = nn::linear(p, total, embed_dim, Default::default());
        assert_eq!(
            total % nheads,
            0,
            "Embedding dim is not divisible by nheads"
        );

        let head_dim = total / nheads;
        Self {
            packed_proj,
            output_proj,
            head_dim,
            nheads,
        }
    }

    pub fn forward(&self, xs: &Tensor, attn_mask: &Tensor) -> Tensor {
        // Expected input shape: [batch; ]

        let result = xs.apply(&self.packed_proj).chunk(3, -1);
        let Ok([query, key, value]) = TryInto::<[Tensor; 3]>::try_into(result) else {
            panic!("chunking failed.")
        };

        // Step 2. Split heads and prepare for SDPA
        // reshape query, key, value to separate by head
        // (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        let query = query
            .unflatten(-1, [self.nheads, self.head_dim])
            .transpose(1, 2);

        // (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        let key = key
            .unflatten(-1, [self.nheads, self.head_dim])
            .transpose(1, 2);

        // (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        let value = value
            .unflatten(-1, [self.nheads, self.head_dim])
            .transpose(1, 2);

        // Step 3. Run SDPA
        // (N, nheads, L_t, E_head)
        let attn_output = Tensor::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            Some(attn_mask),
            0.0f64,
            false,
            None,
            false,
        );

        // (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        let attn_output = attn_output.transpose(1, 2).flatten(-2, -1);

        // Step 4. Apply output projection
        // (N, L_t, E_total) -> (N, L_t, E_out)
        let attn_output = attn_output.apply(&self.output_proj);

        return attn_output;
    }
}
