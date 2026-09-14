"""Application state machine and checkpoints."""

from __future__ import annotations

import pytest

from cvnator.domain.application import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    ApplicationRecord,
    ApplicationState,
    Checkpoint,
    IllegalTransition,
    StateMachine,
    can_transition,
)


def test_happy_path_is_walkable():
    machine = StateMachine()
    for target in (
        ApplicationState.ANALYZING,
        ApplicationState.MATCHED,
        ApplicationState.OPTIMIZING_RESUME,
        ApplicationState.VALIDATING_RESUME,
        ApplicationState.QUEUED,
        ApplicationState.OPENING_APPLICATION,
        ApplicationState.APPLYING,
        ApplicationState.SUBMITTING,
        ApplicationState.SUBMITTED,
    ):
        machine.transition(target)
    assert machine.state is ApplicationState.SUBMITTED
    assert machine.is_terminal
    assert len(machine.history) == 9


def test_cannot_skip_straight_to_submitted():
    machine = StateMachine()
    machine.transition(ApplicationState.ANALYZING)
    machine.transition(ApplicationState.MATCHED)
    with pytest.raises(IllegalTransition):
        machine.transition(ApplicationState.SUBMITTED)


def test_submitted_is_final():
    assert ALLOWED_TRANSITIONS[ApplicationState.SUBMITTED] == frozenset()
    assert not can_transition(ApplicationState.SUBMITTED, ApplicationState.APPLYING)


def test_failed_can_be_requeued_for_a_retry():
    assert can_transition(ApplicationState.FAILED, ApplicationState.QUEUED)


def test_waiting_user_returns_to_applying():
    assert can_transition(ApplicationState.WAITING_USER, ApplicationState.APPLYING)


def test_transitioning_to_the_current_state_is_a_no_op():
    machine = StateMachine(ApplicationState.APPLYING)
    machine.transition(ApplicationState.APPLYING)
    assert machine.history == []


def test_terminal_states_are_exactly_the_four_expected():
    assert frozenset(
        {
            ApplicationState.REJECTED,
            ApplicationState.SUBMITTED,
            ApplicationState.FAILED,
            ApplicationState.ABANDONED,
        }
    ) == TERMINAL_STATES


def test_every_state_has_a_declared_edge_set():
    for state in ApplicationState:
        assert state in ALLOWED_TRANSITIONS, f"{state} has no declared transitions"


def test_last_checkpoint_follows_the_declared_order_not_insertion_order():
    record = ApplicationRecord(
        checkpoints=[Checkpoint.RESUME_UPLOADED, Checkpoint.APPLICATION_OPENED]
    )
    assert record.last_checkpoint() is Checkpoint.RESUME_UPLOADED


def test_last_checkpoint_is_none_when_nothing_reached():
    assert ApplicationRecord().last_checkpoint() is None
