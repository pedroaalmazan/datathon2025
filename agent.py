import os
import uuid

#new
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, height=18, width=20, in_channels=3, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_dim = 32 * height * width
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    import numpy as np
import torch

# Directions as (dx, dy)
DIRECTION_VECTORS = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1, 0),
    "RIGHT": (1,  0),
}

def get_board_size(board):
    h = len(board)
    w = len(board[0]) if h > 0 else 0
    return w, h


#^new

from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

#new
NUM_ACTIONS = 4
MODEL_PATH = "model.pth"
_rl_model = None

def load_model():
    global _rl_model
    model = DQN(height=18, width=20, num_actions=NUM_ACTIONS)
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _rl_model = model
        print(f"[RL] Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[RL] WARNING: could not load model: {e}")
        _rl_model = None

load_model()

def build_obs_from_state(state: dict, player_number: int):
    board = state.get("board")
    if board is None:
        board = GLOBAL_GAME.board.grid

    board_arr = np.array(board, dtype=np.float32)
    h, w = board_arr.shape

    ch_trails = board_arr
    ch_me = np.zeros_like(board_arr, dtype=np.float32)
    ch_opp = np.zeros_like(board_arr, dtype=np.float32)

    my_key = "agent1_trail" if player_number == 1 else "agent2_trail"
    opp_key = "agent2_trail" if player_number == 1 else "agent1_trail"

    my_trail = state.get(my_key) or []
    opp_trail = state.get(opp_key) or []

    if my_trail:
        hx, hy = my_trail[-1]
        ch_me[hy, hx] = 1.0

    if opp_trail:
        ox, oy = opp_trail[-1]
        ch_opp[oy, ox] = 1.0

    obs = np.stack([ch_trails, ch_me, ch_opp], axis=0)
    return obs

#^new

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
            
        if _rl_model is None:
            # Fallback if model didn't load
            # === Fallback logic when model fails or returns bad data ===
            board = state.get("board")
            if board is None:
                board = GLOBAL_GAME.board.grid

            width = len(board[0])
            height = len(board)

            # get our and opponent's positions
            my_trail_key = "agent1_trail" if player_number == 1 else "agent2_trail"
            opp_trail_key = "agent2_trail" if player_number == 1 else "agent1_trail"
            my_trail = state.get(my_trail_key, [])
            opp_trail = state.get(opp_trail_key, [])

            if not my_trail:
                return jsonify({"move": "RIGHT"}), 200

            head_x, head_y = my_trail[-1]
            opp_x, opp_y = opp_trail[-1] if opp_trail else (None, None)

            # direction order preference
            dir_order = ["RIGHT", "UP", "LEFT", "DOWN"]
            DIRECTION_VECTORS = {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0),
            }

            # helper to check if a cell is safe
            def is_safe(x, y):
                return board[y % height][x % width] == 0

            # check proximity to opponent head
            use_boost = False
            if opp_x is not None:
                dist_x = min(abs(head_x - opp_x), width - abs(head_x - opp_x))
                dist_y = min(abs(head_y - opp_y), height - abs(head_y - opp_y))
                if dist_x + dist_y <= 2:
                    use_boost = True

            # find the first safe move
            move_dir = None
            for d in dir_order:
                dx, dy = DIRECTION_VECTORS[d]
                nx = (head_x + dx) % width
                ny = (head_y + dy) % height
                if is_safe(nx, ny):
                    move_dir = d
                    break

            # if no move is safe, just go RIGHT
            if move_dir is None:
                move_dir = "RIGHT"

            # construct move string
            move = f"{move_dir}:BOOST" if use_boost else move_dir
            print(f"[FALLBACK] {move}")

            return jsonify({"move": move}), 200

                    
        else:
            board = state.get("board")
            if board is None:
                board = GLOBAL_GAME.board.grid

            width, height = get_board_size(board)

            # Build observation and get Q-values from the model
            obs = build_obs_from_state(state, player_number)
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # (1, 3, H, W)
            with torch.no_grad():
                q_vals = _rl_model(obs_t).squeeze(0)  # shape: (4,)

            # Find our head position
            my_trail_key = "agent1_trail" if player_number == 1 else "agent2_trail"
            my_trail = state.get(my_trail_key) or []
            if not my_trail:
                move = "RIGHT"  # weird edge case
            else:
                head_x, head_y = my_trail[-1]

                # Map indices to direction strings
                idx_to_dir = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

                # Sort actions by Q-value, best to worst
                indices = list(range(4))
                indices.sort(key=lambda i: float(q_vals[i].item()), reverse=True)

                chosen_dir_str = None

                for idx in indices:
                    dir_str = idx_to_dir[idx]
                    dx, dy = DIRECTION_VECTORS[dir_str]

                    # wraparound next position
                    nx = (head_x + dx) % width
                    ny = (head_y + dy) % height

                    # If the cell is occupied (trail/wall), this is an immediate crash → skip
                    if board[ny][nx] != 0:
                        continue

                    chosen_dir_str = dir_str
                    break

                if chosen_dir_str is None:
                    # All directions lead to immediate collision — doomed, but pick something
                    chosen_dir_str = "RIGHT"

                move = chosen_dir_str

       

    
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
