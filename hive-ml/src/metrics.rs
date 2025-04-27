use std::sync::OnceLock;

use hive_engine::{movement::Move, piece::Color};
use opentelemetry::{
    metrics::{Counter, Gauge, Histogram},
    InstrumentationScope, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::Temporality;

fn scope() -> &'static InstrumentationScope {
    static SCOPE: OnceLock<InstrumentationScope> = OnceLock::new();

    SCOPE.get_or_init(|| {
        InstrumentationScope::builder("training")
            .with_version("1.0")
            .build()
    })
}

pub fn init_meter_provider() -> opentelemetry_sdk::metrics::SdkMeterProvider {
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_temporality(Temporality::Cumulative)
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary) //can be changed to `Protocol::HttpJson` to export in JSON format
        .build()
        .expect("Failed to create metric exporter");

    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_periodic_exporter(exporter)
        .build();

    opentelemetry::global::set_meter_provider(provider.clone());
    provider
}

pub fn record_epoch(epoch: usize) {
    static METRIC: OnceLock<Gauge<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_gauge("epoch")
            .with_description("a counter for every time the opponent version is upgraded")
            .with_unit("times")
            .build()
    });

    metric.record(epoch as u64, &[]);
}

pub fn record_training_start_time() {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("train_start_time")
            .with_description("training start time")
            .build()
    });

    metric.record(chrono::Utc::now().timestamp() as f64, &[]);
}

pub fn record_training_status(is_training: bool) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("is_training")
            .with_description("Is the model currently training")
            .build()
    });

    metric.record(if is_training { 1.0 } else { 0.0 }, &[]);
}

pub fn increment_leveled_up_opponent() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_counter("opponent_version_increased_total")
            .with_description("a counter for every time the opponent version is upgraded")
            .with_unit("times")
            .build()
    });

    metric.add(1, &[]);
}

pub fn increment_games_finished(
    model_color: Color,
    winner: Option<Color>,
    stalled: bool,
    model_name: &str,
    opponent_name: &str,
) {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_counter("games_finished_total")
            .with_description("a counter for games finished.")
            .with_unit("games")
            .build()
    });

    let (color, opponent_color) = match model_color {
        Color::Black => ("black", "white"),
        Color::White => ("white", "black"),
    };

    let winner = match winner {
        Some(Color::Black) => "black",
        Some(Color::White) => "white",
        None if stalled => "stalled",
        None => "draw",
    };

    metric.add(
        1,
        &[
            KeyValue::new("model_color", color),
            KeyValue::new("winner", winner),
            KeyValue::new("model_name", model_name.to_string()),
            KeyValue::new("opponent_name", opponent_name.to_string()),
        ],
    );

    metric.add(
        1,
        &[
            KeyValue::new("model_color", opponent_color),
            KeyValue::new("winner", winner),
            KeyValue::new("model_name", opponent_name.to_string()),
            KeyValue::new("opponent_name", model_name.to_string()),
        ],
    );
}

pub fn increment_move_made(mv: Move, model_color: Color, model_name: &str, opponent_name: &str) {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_counter("moves_made_total")
            .with_description("a counter for every move the model makes")
            .build()
    });

    let mv_type = match mv {
        Move::MovePiece { .. } => "move_piece",
        Move::PlacePiece { .. } => "place_piece",
        Move::Pass => "pass",
    };

    let piece = match mv {
        Move::MovePiece { piece, .. } | Move::PlacePiece { piece, .. } => match piece.role {
            hive_engine::piece::Insect::Grasshopper => "grasshopper",
            hive_engine::piece::Insect::QueenBee => "queen_bee",
            hive_engine::piece::Insect::Beetle => "beetle",
            hive_engine::piece::Insect::Spider => "spider",
            hive_engine::piece::Insect::SoldierAnt => "soldier_ant",
        },
        Move::Pass => "none",
    };

    let color = match model_color {
        Color::Black => "black",
        Color::White => "white",
    };

    metric.add(
        1,
        &[
            KeyValue::new("move_type", mv_type),
            KeyValue::new("piece", piece),
            KeyValue::new("model_color", color),
            KeyValue::new("model_name", model_name.to_string()),
            KeyValue::new("opponent_name", opponent_name.to_string()),
        ],
    );
}

pub fn record_game_turns(
    game_turns: usize,
    model_color: Color,
    model_name: &str,
    opponent_name: &str,
) {
    static METRIC: OnceLock<Histogram<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_histogram("turns_played")
            .with_description("the number of turns in a game.")
            .with_unit("games")
            .with_boundaries((0..151).map(|x| 20.0 * x as f64).collect::<Vec<_>>())
            .build()
    });

    let (color, opponent_color) = match model_color {
        Color::Black => ("black", "white"),
        Color::White => ("white", "black"),
    };

    metric.record(
        game_turns as u64,
        &[
            KeyValue::new("model_color", color),
            KeyValue::new("model_name", model_name.to_string()),
            KeyValue::new("opponent_name", opponent_name.to_string()),
        ],
    );

    metric.record(
        game_turns as u64,
        &[
            KeyValue::new("model_color", opponent_color),
            KeyValue::new("model_name", opponent_name.to_string()),
            KeyValue::new("opponent_name", model_name.to_string()),
        ],
    );
}

pub fn record_game_duration(
    game_duration: f64,
    model_color: Color,
    model_name: &str,
    opponent_name: &str,
) {
    static METRIC: OnceLock<Histogram<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        let boundaries = (0..400).map(|i| 0.05 * i as f64).collect::<Vec<_>>();

        meter
            .f64_histogram("game_duration")
            .with_description("the wall clock duration of a game.")
            .with_unit("s")
            .with_boundaries(boundaries)
            .build()
    });

    let (color, opponent_color) = match model_color {
        Color::Black => ("black", "white"),
        Color::White => ("white", "black"),
    };

    metric.record(
        game_duration,
        &[
            KeyValue::new("model_color", color),
            KeyValue::new("model_name", model_name.to_string()),
            KeyValue::new("opponent_name", opponent_name.to_string()),
        ],
    );

    metric.record(
        game_duration,
        &[
            KeyValue::new("model_color", opponent_color),
            KeyValue::new("model_name", opponent_name.to_string()),
            KeyValue::new("opponent_name", model_name.to_string()),
        ],
    );
}

pub fn record_frame_buffer_count(frame_count: usize) {
    static METRIC: OnceLock<Gauge<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_gauge("current_frame_buffer_count")
            .with_description("the number of frames in the buffer")
            .build()
    });

    metric.record(frame_count as u64, &[]);
}

pub fn record_training_duration(duration: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("training_duration")
            .with_description("the wall clock duration of a game.")
            .with_unit("s")
            .build()
    });

    metric.record(duration, &[]);
}

pub fn record_data_generation_duration(duration: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("training_data_generation_duration")
            .with_description("the wall clock time to generate the training data.")
            .with_unit("s")
            .build()
    });

    metric.record(duration, &[]);
}

pub fn record_learning_rate(lr: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("learning_rate")
            .with_description("the learning rate used during training.")
            .build()
    });

    metric.record(lr, &[]);
}

pub fn record_entropy_loss_scale(lr: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("entropy_loss_scale")
            .with_description("the scale factor for entropy loss")
            .build()
    });

    metric.record(lr, &[]);
}

pub fn record_value_mse(mse: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("value_fn_mse")
            .with_description("the mse of the value function during a train loop")
            .build()
    });

    metric.record(mse, &[]);
}

pub fn record_training_batches(batch_count: usize) {
    static METRIC: OnceLock<Gauge<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_gauge("training_batch_count")
            .with_description("the number of iterations run during a training loop")
            .build()
    });

    metric.record(batch_count as u64, &[]);
}

pub fn record_win_rate_vs_initial(win_rate: f64, model_name: &str) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("win_rate_vs_random")
            .with_description("a counter for games won by model.")
            .build()
    });

    metric.record(
        win_rate,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
}

pub fn record_weighted_game_length(length: f64, model_name: &str) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("weighted_game_length")
            .with_description("the exponential moving average of game lengths used for sampling frames")
            .build()
    });

    metric.record(
        length,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
}

pub fn record_policy_minibatch_statistics(
    value_loss: f64,
    policy_loss: f64,
    entropy_loss: f64,
    pi_ratio: f64,
    pi_adv: f64,
    approximate_kl: f64,
    model_name: &str,
) {
    static TRAINING_ITERATIONS_COUNT_TOTAL: OnceLock<Counter<u64>> = OnceLock::new();
    static TRAINING_VALUE_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_POLICY_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_ENTROPY_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_PI_RATIO_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_PI_ADV_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_APPROXIMATE_KL_GAUGE: OnceLock<Gauge<f64>> = OnceLock::new();

    let training_iterations_count_total = TRAINING_ITERATIONS_COUNT_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_counter("training_iterations_count_total")
            .with_description("a counter for the number of training iterations run.")
            .build()
    });

    let training_value_loss_total = TRAINING_VALUE_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_value_loss_total")
            .with_description("a counter for the total value loss during training.")
            .build()
    });

    let training_policy_loss_total = TRAINING_POLICY_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_policy_loss_total")
            .with_description("a counter for the total policy loss during training.")
            .build()
    });

    let training_entropy_loss_total = TRAINING_ENTROPY_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_entropy_loss_total")
            .with_description("a counter for the total entropy loss during training.")
            .build()
    });

    let training_pi_ratio_total = TRAINING_PI_RATIO_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_pi_ratio_total")
            .with_description("a counter for the total ratio between new and old policy.")
            .build()
    });

    let training_pi_adv_total = TRAINING_PI_ADV_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_pi_adv_total")
            .with_description("a counter for the total advantage.")
            .build()
    });

    let training_approximate_kl_gauge = TRAINING_APPROXIMATE_KL_GAUGE.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_gauge("training_approximate_kl")
            .with_description("the approximate KL divergence during training")
            .build()
    });

    training_iterations_count_total.add(1, &[KeyValue::new("model_name", model_name.to_string())]);
    training_value_loss_total.add(
        value_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_policy_loss_total.add(
        policy_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_entropy_loss_total.add(
        entropy_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_approximate_kl_gauge.record(
        approximate_kl,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_pi_ratio_total.add(
        pi_ratio,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_pi_adv_total.add(
        pi_adv,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
}

pub fn record_auxiliary_minibatch_statistics(
    value_loss: f64,
    soft_value_loss: f64,
    kl_loss: f64,
    model_name: &str,
) {
    static TRAINING_ITERATIONS_COUNT_TOTAL: OnceLock<Counter<u64>> = OnceLock::new();
    static TRAINING_VALUE_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_SOFT_VALUE_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();
    static TRAINING_KL_LOSS_TOTAL: OnceLock<Counter<f64>> = OnceLock::new();

    let training_iterations_count_total = TRAINING_ITERATIONS_COUNT_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .u64_counter("training_auxiliary_iterations_count_total")
            .with_description("a counter for the number of training iterations run.")
            .build()
    });

    let training_value_loss_total = TRAINING_VALUE_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_auxiliary_value_loss_total")
            .with_description("a counter for the total value loss during training.")
            .build()
    });

    let training_soft_value_loss_total = TRAINING_SOFT_VALUE_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_auxiliary_soft_value_loss_total")
            .with_description("a counter for the total policy loss during training.")
            .build()
    });

    let training_kl_loss_total = TRAINING_KL_LOSS_TOTAL.get_or_init(|| {
        let meter = opentelemetry::global::meter_with_scope(scope().clone());

        meter
            .f64_counter("training_auxiliary_kl_loss_total")
            .with_description("a counter for the total entropy loss during training.")
            .build()
    });

    training_iterations_count_total.add(1, &[KeyValue::new("model_name", model_name.to_string())]);
    training_value_loss_total.add(
        value_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_soft_value_loss_total.add(
        soft_value_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
    training_kl_loss_total.add(
        kl_loss,
        &[KeyValue::new("model_name", model_name.to_string())],
    );
}
