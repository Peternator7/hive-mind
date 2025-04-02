use std::f64;

use tch::{nn::{self, Embedding}, Tensor};

use crate::hypers::INPUT_ENCODED_DIMS;

pub struct HiveTransformerModel {
    pub device: tch::Device,
    emb: Embedding,
    initial_layer: MultiHeadSelfAttention,
    repeated_layers: Vec<MultiHeadSelfAttention>,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
    pub train_mode: std::sync::atomic::AtomicBool,
}

impl HiveTransformerModel {
    pub fn new(p: &nn::Path) -> Self {
        let embedded_size = 12;
        let attn_size = 6 + embedded_size;

        let emb = nn::embedding(p, 1 + INPUT_ENCODED_DIMS as i64, embedded_size, Default::default());
        let initial_layer = MultiHeadSelfAttention::new(&p.sub("a0"), attn_size, 96, 8);

        let mut repeated_layers = Vec::new();
        for i in 1..5 {
            let s = format!("a{}", i);
            let p = p / s;
            repeated_layers.push(MultiHeadSelfAttention::new(&p, attn_size, 96, 8));
        }

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", attn_size, 1, Default::default()))
            .add_fn(|xs| xs.sigmoid() - 0.5);

        let policy_layer = nn::seq().add(nn::linear(
            p / "al",
            attn_size,
            crate::hypers::OUTPUT_LENGTH as i64,
            Default::default(),
        ));

        Self {
            emb,
            initial_layer,
            repeated_layers,
            policy_layer,
            value_layer,
            train_mode: false.into(),
            device: p.device(),
        }
    }

    pub fn value_policy(&self, pieces: &Tensor, locations: &Tensor, seq_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        let t = self.shared_layers(pieces, locations, seq_mask);
        (t.apply(&self.value_layer), t.apply(&self.policy_layer))
    }

    pub fn policy(&self, pieces: &Tensor, locations: &Tensor, seq_mask: Option<&Tensor>) -> Tensor {
        let t = self.shared_layers(pieces,locations, seq_mask);
        t.apply(&self.policy_layer)
    }
    
    fn shared_layers(&self, pieces: &Tensor, locations: &Tensor, seq_mask: Option<&Tensor>) -> Tensor {
        let pieces = pieces.apply(&self.emb);
        
        const SCALE: f64 = 2.0 * f64::consts::PI / 64.0;

        let sin_loc = (SCALE * locations).sin();
        let cos_loc = (SCALE * locations).cos();

        let game_state = Tensor::cat(&[pieces, sin_loc, cos_loc], -1);

        let mut output = &game_state + self.initial_layer.forward(&game_state, seq_mask);
        for attn in &self.repeated_layers {
            output = &output + attn.forward(&output, seq_mask);
        }

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
        let packed_proj = nn::linear(p / "i1", embed_dim, total * 3, Default::default());
        let output_proj = nn::linear(p / "o1", total, embed_dim, Default::default());
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

    pub fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Tensor {
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
            attn_mask,
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
