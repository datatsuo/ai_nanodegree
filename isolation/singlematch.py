#from tournament import play_matches
from collections import namedtuple
import random

from isolation import Board
from sample_players import (RandomPlayer, open_move_score,improved_score, center_score)
from game_agent import (MinimaxPlayer, AlphaBetaPlayer, custom_score, custom_score_2, custom_score_3)

Agent = namedtuple("Agent", ["player", "name"])

# introduce the test agent and cpu agent
test_agent = Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved")
cpu_agent = Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved2")
# cpu_agent = Agent(RandomPlayer(), "Random")

#NUM_MATCHES = 1  # number of matches against each opponent

TIME_LIMIT = 150  # number of milliseconds before timeout

game = Board(cpu_agent.player, test_agent.player)

# initialize all games with a random move and response
for _ in range(2):
    move = random.choice(game.get_legal_moves())
    game.apply_move(move)

winner, history, termination = game.play(time_limit=TIME_LIMIT)

print("winner:", winner)
print("history:", history)
print("termination:", termination)
# note that winner will be displayed in green while the lowser is in red
