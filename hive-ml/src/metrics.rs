use std::sync::OnceLock;

use hive_engine::movement::Move;
use opentelemetry::{
    metrics::{Counter, Gauge, Histogram},
    InstrumentationScope, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::Temporality;

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
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_gauge("epoch")
            .with_description("a counter for every time the opponent version is upgraded")
            .with_unit("times")
            .build()
    });

    metric.record(epoch as u64, &[]);
}

pub fn increment_leveled_up_opponent() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_counter("opponent_version_increased_total")
            .with_description("a counter for every time the opponent version is upgraded")
            .with_unit("times")
            .build()
    });

    metric.add(1, &[]);
}

pub fn increment_games_played() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_counter("games_played_total")
            .with_description("a counter for games started.")
            .with_unit("games")
            .build()
    });

    metric.add(1, &[]);
}

pub fn increment_move_made(mv: Move) {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

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

    metric.add(
        1,
        &[
            KeyValue::new("move_type", mv_type),
            KeyValue::new("piece", piece),
        ],
    );
}

pub fn increment_games_finished() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_counter("games_finished_total")
            .with_description("a counter for games finished.")
            .with_unit("games")
            .build()
    });

    metric.add(1, &[]);
}

pub fn increment_white_side_won() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_counter("white_side_won_total")
            .with_description("a counter for games won by the player with the white pieces.")
            .with_unit("games")
            .build()
    });

    metric.add(1, &[]);
}

pub fn increment_model_won() {
    static METRIC: OnceLock<Counter<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_counter("model_won_total")
            .with_description("a counter for games won by model.")
            .with_unit("games")
            .build()
    });

    metric.add(1, &[]);
}

pub fn record_game_turns(game_turns: usize) {
    static METRIC: OnceLock<Histogram<u64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_histogram("turns_played")
            .with_description("the number of turns in a game.")
            .with_unit("games")
            .with_boundaries((0..100).map(|x| 20.0 * x as f64).collect::<Vec<_>>())
            .build()
    });

    metric.record(game_turns as u64, &[]);
}

pub fn record_game_duration(game_duration: f64) {
    static METRIC: OnceLock<Histogram<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        let boundaries = (0..100).map(|i| 0.05 * i as f64).collect::<Vec<_>>();

        meter
            .f64_histogram("game_duration")
            .with_description("the wall clock duration of a game.")
            .with_unit("s")
            .with_boundaries(boundaries)
            .build()
    });

    metric.record(game_duration, &[]);
}

pub fn record_training_duration(duration: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

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
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

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
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .f64_gauge("learning_rate")
            .with_description("the learning rate used during training.")
            .build()
    });

    metric.record(lr, &[]);
}

pub fn record_value_mse(mse: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

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
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .u64_gauge("training_batch_count")
            .with_description("the number of iterations run during a training loop")
            .build()
    });

    metric.record(batch_count as u64, &[]);
}

pub fn record_win_rate_vs_initial(win_rate: f64) {
    static METRIC: OnceLock<Gauge<f64>> = OnceLock::new();

    let metric = METRIC.get_or_init(|| {
        let scope = InstrumentationScope::builder("training")
            .with_version("1.0")
            .build();

        let meter = opentelemetry::global::meter_with_scope(scope);

        meter
            .f64_gauge("win_rate_vs_random")
            .with_description("a counter for games won by model.")
            .build()
    });

    metric.record(win_rate, &[]);
}
