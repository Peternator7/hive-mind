use crate::hypers::{INPUT_ENCODED_DIMS, OUTPUT_LENGTH};
use tch::{nn, Tensor};

pub struct HiveTransformerModel {
    pub device: tch::Device,
    t1: MultiHeadSelfAttention,
    t2: MultiHeadSelfAttention,
    t3: MultiHeadSelfAttention,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
}

impl HiveTransformerModel {
    pub fn new(p: &nn::Path) -> Self {
        let i = 6 + INPUT_ENCODED_DIMS as i64;

        let t1 = MultiHeadSelfAttention::new(p, i, 1024, 8);
        let t2 = MultiHeadSelfAttention::new(p, i, 1024, 8);
        let t3 = MultiHeadSelfAttention::new(p, i, 1024, 8);

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", 36, 1, Default::default()))
            .add_fn(|xs| xs.sigmoid() - 0.5);

        let policy_layer = nn::seq().add(nn::linear(
            p / "al",
            36,
            OUTPUT_LENGTH as i64,
            Default::default(),
        ));

        Self {
            t1,
            t2,
            t3,
            policy_layer,
            value_layer,
            device: p.device(),
        }
    }

    pub fn value_policy(&self, game_state: &Tensor, seq_mask: &Tensor) -> (Tensor, Tensor) {
        let t = self.shared_layers(game_state, seq_mask);
        (t.apply(&self.value_layer), t.apply(&self.policy_layer))
    }

    pub fn policy(&self, game_state: &Tensor, seq_mask: &Tensor) -> Tensor {
        let t = self.shared_layers(game_state, seq_mask);
        t.apply(&self.policy_layer)
    }

    fn shared_layers(&self, game_state: &Tensor, seq_mask: &Tensor) -> Tensor {
        let output = game_state + self.t1.forward(game_state, seq_mask);
        let output = &output + self.t2.forward(&output, seq_mask);
        let output = &output + self.t3.forward(&output, seq_mask);

        // shape is [batch, seq, embed]
        output.index(&[None, Some(Tensor::from(0))])
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
