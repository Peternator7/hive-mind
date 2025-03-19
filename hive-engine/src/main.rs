use std::{
    future::Future,
    task::{Context, Waker},
};

use hive_engine::game::Game;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fut = run_game();
    match Box::pin(fut)
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        std::task::Poll::Ready(output) => {
            let _ = output?;
        }
        std::task::Poll::Pending => {}
    }

    Ok(())
}

async fn run_game() -> Result<(), Box<dyn std::error::Error>> {
    let mut game = Game::new();

    let mut white_agent = hive_engine::ab_agent::Agent::new(6);
    let mut black_agent = hive_engine::ab_agent::Agent::new(6);

    while game.is_game_is_over().is_none() {
        println!("Turn: {}", game.turn());

        let mv = white_agent.determine_best_move(&game).await?;
        game.make_move(mv)?;
        if game.is_game_is_over().is_some() {
            break;
        }

        let mv = black_agent.determine_best_move(&game).await?;
        game.make_move(mv)?;
    }

    println!("Result: {:?}", game.is_game_is_over());
    Ok(())
}
