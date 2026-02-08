import pyspiel
import deckgym
import numpy as np

class DeckGymState(pyspiel.State):
    def __init__(self, game, deck_id_1, deck_id_2, seed=None, rust_game=None):
        super().__init__(game)
        self._deck_id_1 = deck_id_1
        self._deck_id_2 = deck_id_2
        self._seed = seed
        self._pending_stochastic_action = None

        if rust_game:
            self.rust_game = rust_game
        else:
            self.rust_game = deckgym.PyGameState(deck_id_1, deck_id_2, seed)

    def current_player(self):
        if self._pending_stochastic_action is not None:
            return pyspiel.PlayerId.CHANCE
        if self.rust_game.get_state().is_game_over():
            return pyspiel.PlayerId.TERMINAL
        return self.rust_game.get_state().current_player

    def _legal_actions(self, player):
        if self.is_chance_node():
            # Return outcome indices 0..N-1
            # We stored (action_id, probs) in _pending_stochastic_action
            probs = self._pending_stochastic_action[1]
            return list(range(len(probs)))
        else:
            return self.rust_game.legal_actions()

    def chance_outcomes(self):
        if self.is_chance_node():
            probs = self._pending_stochastic_action[1]
            return list(enumerate(probs))
        return []

    def _apply_action(self, action_id):
        if self.is_chance_node():
            # Apply the outcome
            outcome_idx = action_id
            real_action_id, _ = self._pending_stochastic_action
            self.rust_game.apply_action_outcome(real_action_id, outcome_idx)
            self._pending_stochastic_action = None
        else:
            # Player action
            # Check if this action is stochastic
            probs = self.rust_game.get_action_probabilities(action_id)
            if len(probs) > 1:
                # Stochastic action: transition to chance node
                self._pending_stochastic_action = (action_id, probs)
            else:
                # Deterministic action
                self.rust_game.step_with_id(action_id)

    def _action_to_string(self, player, action_id):
        if player == pyspiel.PlayerId.CHANCE:
            return f"outcome_{action_id}"
        else:
            return deckgym.PyGameState.action_name(action_id)

    def is_terminal(self):
        return self.rust_game.get_state().is_game_over()

    def returns(self):
        if self.is_terminal():
            outcome = self.rust_game.get_state().winner
            if outcome is None:
                return [0.0, 0.0]

            if outcome.is_tie:
                return [0.0, 0.0]

            if outcome.winner is not None:
                if outcome.winner == 0:
                    return [1.0, -1.0]
                else:
                    return [-1.0, 1.0]
        return [0.0, 0.0]

    def observation_tensor(self, player):
        # Flatten the list returned by encode_observation
        return self.rust_game.encode_observation(player_id=player, public_only=True)

    def information_state_tensor(self, player):
        return self.rust_game.encode_observation(player_id=player, public_only=False)

    def observation_string(self, player):
        return self.rust_game.get_state().debug_string()

    def information_state_string(self, player):
        return self.rust_game.get_state().debug_string()

    def clone(self):
        # We need to clone the rust game state
        # PyGameState has a clone() method that takes a Python context implicitly via pyo3
        # But in Python we just call .clone()
        # Wait, the signature in rust is `pub fn clone(&self, py: Python) -> PyResult<Self>`
        # In Python, `py` is handled automatically.
        # However, `PyGameState.clone` returns a new `PyGameState`.
        new_rust_game = self.rust_game.clone()
        new_state = DeckGymState(
            self.get_game(),
            self._deck_id_1,
            self._deck_id_2,
            self._seed,
            rust_game=new_rust_game
        )
        new_state._pending_stochastic_action = self._pending_stochastic_action
        return new_state

    def __str__(self):
        return str(self.rust_game.get_state())

    def __repr__(self):
        return repr(self.rust_game.get_state())

    def __deepcopy__(self, memo):
        return self.clone()
