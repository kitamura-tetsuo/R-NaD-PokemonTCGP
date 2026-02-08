import pyspiel
import deckgym
from .state import DeckGymState

_GAME_TYPE = pyspiel.GameType(
    short_name="deckgym_ptcgp",
    long_name="DeckGym Pokemon TCG Pocket",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "deck_id_1": "deckgym-core/example_decks/mewtwoex.txt",
        "deck_id_2": "deckgym-core/example_decks/mewtwoex.txt",
        "seed": 0
    }
)

_GAME_INFO = pyspiel.GameInfo(
    deckgym.PyGameState.get_action_space_size(),
    100,
    2,
    -1.0,
    1.0,
    0.0,
    1000
)

class DeckGymGame(pyspiel.Game):
    def __init__(self, params=None):
        if params is None:
            params = {}

        self._deck_id_1 = params.get("deck_id_1", "default_deck")
        self._deck_id_2 = params.get("deck_id_2", "default_deck")
        self._seed = params.get("seed", None)

        try:
            dummy_state = deckgym.PyGameState(self._deck_id_1, self._deck_id_2, self._seed)
            self._obs_shape = [len(dummy_state.encode_observation())]
        except Exception:
             self._obs_shape = [0]

        super().__init__(_GAME_TYPE, _GAME_INFO, params)

    def new_initial_state(self):
        # Pass deck configurations stored in self.params or self.deck_ids
        return DeckGymState(self, self._deck_id_1, self._deck_id_2, self._seed)

    def num_distinct_actions(self):
        return deckgym.PyGameState.get_action_space_size()

    def max_chance_outcomes(self):
        return 100

    def num_players(self):
        return 2

    def min_utility(self):
        return -1.0

    def max_utility(self):
        return 1.0

    def utility_sum(self):
        return 0.0

    def information_state_tensor_shape(self):
        return self._obs_shape

    def observation_tensor_shape(self):
        return self._obs_shape

pyspiel.register_game(_GAME_TYPE, DeckGymGame)
