import sys
sys.path.append('../')
from utils import get_distance, get_center

class PlayerBallAssigner:
    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center(ball_bbox)

        min_distance = 9999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = get_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = get_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player