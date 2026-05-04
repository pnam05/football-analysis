import cv2
import sys
sys.path.append('../')
from utils import get_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.window_size = 12
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for obj, obj_track in tracks.items():
            if obj == 'ball' or obj == 'referees':
                continue
            number_of_frames = len(obj_track)
            for frame_num in range(0, number_of_frames, self.window_size):
                last_frame = min(frame_num+self.window_size, number_of_frames-1)

                for track_id, _ in obj_track[frame_num].items():
                    if track_id not in obj_track[last_frame]:
                        continue

                    start_position = obj_track[frame_num][track_id]['position_transformed']
                    end_position = obj_track[last_frame][track_id]['position_transformed']
                    if start_position is None or end_position is None:
                        continue

                    distance_covered = get_distance(start_position, end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meters_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meters_per_second*3.6

                    if obj not in total_distance:
                        total_distance[obj] = {}

                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0

                    total_distance[obj][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        tracks[obj][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        out_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == 'ball' or obj == 'referees':
                    continue
                
                for _, track_info in obj_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, (0.5), (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, (0.5), (0,0,0), 2)

            out_frames.append(frame)

        return out_frames