use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use shengji_core::interactive::InteractiveGame;
use shengji_mechanics::types::PlayerID;
use shengji_types::GameMessage;
use storage::Storage;

use crate::serving_types::VersionedGame;

pub async fn try_read_file<M: serde::de::DeserializeOwned>(path: &'_ str) -> Result<M, io::Error> {
    let mut f = tokio::fs::File::open(path).await?;
    let mut data = vec![];
    f.read_to_end(&mut data).await?;
    Ok(serde_json::from_slice(&data)?)
}

pub async fn try_read_file_opt<M: serde::de::DeserializeOwned>(
    path: &'_ str,
) -> Result<Option<M>, io::Error> {
    match try_read_file(path).await {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

pub async fn write_state_to_disk<M: serde::ser::Serialize>(
    path: &'_ str,
    state: &HashMap<String, M>,
) -> std::io::Result<()> {
    let mut f = tokio::fs::File::create(path).await?;
    let json = serde_json::to_vec(state)?;
    f.write_all(&json).await?;
    f.sync_all().await?;

    Ok(())
}

pub async fn execute_immutable_operation<S, E, F>(
    ws_id: usize,
    room_name: &str,
    backend_storage: S,
    operation: F,
    action_description: &'static str,
) -> bool
where
    S: Storage<VersionedGame, E>,
    E: Send,
    F: FnOnce(&InteractiveGame, u64) -> Result<Vec<GameMessage>, anyhow::Error> + Send + 'static,
{
    let room_name_ = room_name.as_bytes().to_vec();

    let res = backend_storage
        .clone()
        .execute_operation_with_messages::<EitherError<E>, _>(
            room_name_.clone(),
            move |versioned_game| {
                let g = InteractiveGame::new_from_state(versioned_game.game);
                let msgs = operation(&g, versioned_game.monotonic_id).map_err(EitherError::E2)?;
                Ok((
                    VersionedGame {
                        game: g.into_state(),
                        room_name: versioned_game.room_name,
                        monotonic_id: versioned_game.monotonic_id,
                        associated_websockets: versioned_game.associated_websockets,
                    },
                    msgs,
                ))
            },
        )
        .await;
    match res {
        Ok(_) => true,
        Err(EitherError::E(_)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            false
        }
        Err(EitherError::E2(msg)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}: {msg}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            false
        }
    }
}

pub async fn execute_operation<S, E, F>(
    ws_id: usize,
    room_name: &str,
    backend_storage: S,
    operation: F,
    action_description: &'static str,
) -> bool
where
    S: Storage<VersionedGame, E>,
    E: Send,
    F: FnOnce(
            &mut InteractiveGame,
            u64,
            &mut HashMap<PlayerID, Vec<usize>>,
        ) -> Result<Vec<GameMessage>, anyhow::Error>
        + Send
        + 'static,
{
    let room_name_ = room_name.as_bytes().to_vec();

    let res = backend_storage
        .clone()
        .execute_operation_with_messages::<EitherError<E>, _>(
            room_name_.clone(),
            move |versioned_game| {
                let mut g = InteractiveGame::new_from_state(versioned_game.game);
                let mut associated_websockets = versioned_game.associated_websockets;
                let mut msgs = operation(
                    &mut g,
                    versioned_game.monotonic_id,
                    &mut associated_websockets,
                )
                .map_err(EitherError::E2)?;
                let game = g.into_state();
                msgs.push(GameMessage::State {
                    state: game.clone(),
                });
                Ok((
                    VersionedGame {
                        room_name: versioned_game.room_name,
                        game,
                        associated_websockets,
                        monotonic_id: versioned_game.monotonic_id + 1,
                    },
                    msgs,
                ))
            },
        )
        .await;
    match res {
        Ok(_) => true,
        Err(EitherError::E(_)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            false
        }
        Err(EitherError::E2(msg)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}: {msg}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            false
        }
    }
}

/// Outcome of an operation that is conditional on the room still having the
/// version from which an expensive result was computed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VersionedOperationOutcome {
    /// The expected version matched and the operation was applied.
    Applied,
    /// The room changed after the caller took its snapshot.  Nothing was
    /// mutated or published; the caller should take a fresh snapshot.
    Stale,
    /// Storage or the operation itself failed.  An error was sent to the
    /// initiating subscriber, matching [`execute_operation`].
    Failed,
}

/// Apply a mutation only if the room is still at `expected_version`.
///
/// Bot search runs outside the storage lock.  Checking only that the same seat
/// is still responsible is insufficient: a takeback or exchange operation can
/// change the position while leaving the same actor and broad action kind.  A
/// monotonic compare-and-set makes the computed move belong to exactly the
/// snapshot it was planned from.  A stale result is a true no-op: it does not
/// bump the version and does not publish a duplicate State message.
pub async fn execute_operation_at_version<S, E, F>(
    ws_id: usize,
    room_name: &str,
    backend_storage: S,
    expected_version: u64,
    operation: F,
    action_description: &'static str,
) -> VersionedOperationOutcome
where
    S: Storage<VersionedGame, E>,
    E: Send,
    F: FnOnce(
            &mut InteractiveGame,
            &mut HashMap<PlayerID, Vec<usize>>,
        ) -> Result<Option<Vec<GameMessage>>, anyhow::Error>
        + Send
        + 'static,
{
    let room_name_ = room_name.as_bytes().to_vec();
    let applied = Arc::new(AtomicBool::new(false));
    let applied_in_operation = Arc::clone(&applied);

    let res = backend_storage
        .clone()
        .execute_operation_with_messages::<EitherError<E>, _>(
            room_name_.clone(),
            move |versioned_game| {
                if versioned_game.monotonic_id != expected_version {
                    // Return the state byte-for-byte unchanged.  Storage only
                    // writes when the version differs, and there are no messages
                    // to publish, so this is a genuine stale-result drop.
                    return Ok((versioned_game, vec![]));
                }

                let mut game = InteractiveGame::new_from_state(versioned_game.game);
                let mut associated_websockets = versioned_game.associated_websockets;
                let messages =
                    operation(&mut game, &mut associated_websockets).map_err(EitherError::E2)?;
                let Some(mut messages) = messages else {
                    return Ok((
                        VersionedGame {
                            room_name: versioned_game.room_name,
                            game: game.into_state(),
                            associated_websockets,
                            monotonic_id: versioned_game.monotonic_id,
                        },
                        vec![],
                    ));
                };
                let game = game.into_state();
                messages.push(GameMessage::State {
                    state: game.clone(),
                });
                applied_in_operation.store(true, Ordering::Release);
                Ok((
                    VersionedGame {
                        room_name: versioned_game.room_name,
                        game,
                        associated_websockets,
                        monotonic_id: versioned_game.monotonic_id + 1,
                    },
                    messages,
                ))
            },
        )
        .await;

    match res {
        Ok(_) if applied.load(Ordering::Acquire) => VersionedOperationOutcome::Applied,
        Ok(_) => VersionedOperationOutcome::Stale,
        Err(EitherError::E(_)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            VersionedOperationOutcome::Failed
        }
        Err(EitherError::E2(msg)) => {
            let err = GameMessage::Error(format!("Failed to {action_description}: {msg}"));
            let _ = backend_storage
                .publish_to_single_subscriber(room_name_, ws_id, err)
                .await;
            VersionedOperationOutcome::Failed
        }
    }
}

enum EitherError<E> {
    E(E),
    E2(anyhow::Error),
}
impl<E> From<E> for EitherError<E> {
    fn from(e: E) -> Self {
        EitherError::E(e)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use slog::{o, Drain, Logger};
    use storage::{HashMapStorage, Storage};

    use super::{execute_operation_at_version, VersionedOperationOutcome};
    use crate::serving_types::VersionedGame;

    fn test_storage() -> HashMapStorage<VersionedGame> {
        let logger = Logger::root(slog::Discard.fuse(), o!());
        HashMapStorage::new(logger)
    }

    fn state(version: u64) -> VersionedGame {
        VersionedGame {
            room_name: b"room".to_vec(),
            game: shengji_core::game_state::GameState::Initialize(
                shengji_core::game_state::initialize_phase::InitializePhase::new(),
            ),
            associated_websockets: HashMap::new(),
            monotonic_id: version,
        }
    }

    #[tokio::test]
    async fn versioned_operation_is_a_true_noop_when_stale_or_declined() {
        let storage = test_storage();
        storage.clone().put(state(3)).await.unwrap();

        let called = Arc::new(AtomicBool::new(false));
        let called_in_op = Arc::clone(&called);
        let outcome = execute_operation_at_version(
            1,
            "room",
            storage.clone(),
            2,
            move |_, _| {
                called_in_op.store(true, Ordering::Release);
                Ok(Some(vec![]))
            },
            "test operation",
        )
        .await;
        assert_eq!(outcome, VersionedOperationOutcome::Stale);
        assert!(!called.load(Ordering::Acquire));
        assert_eq!(
            storage
                .clone()
                .get(b"room".to_vec())
                .await
                .unwrap()
                .monotonic_id,
            3
        );

        let outcome = execute_operation_at_version(
            1,
            "room",
            storage.clone(),
            3,
            |_, _| Ok(None),
            "test operation",
        )
        .await;
        assert_eq!(outcome, VersionedOperationOutcome::Stale);
        assert_eq!(storage.get(b"room".to_vec()).await.unwrap().monotonic_id, 3);
    }
}
