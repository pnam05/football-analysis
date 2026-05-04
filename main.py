from ultralytics import YOLO
from utils import read_video, save_video
from trackers import Tracker
from team_assigners import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigners import PostProcessor

def process_video(input_path, output_path):
    frames = read_video(input_path)
    
    tracker = Tracker('best.pt')
    tracks = tracker.get_obj_trackers(frames, read_from_stub=True, stub_path='stubs/track_stub.pkl')

    tracker.add_position_to_track(tracks)

    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
    )
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    post_processor = PostProcessor(team_assigner)
    tracks = post_processor.merge_fragmented_tracks(tracks)
    tracks = post_processor.fix_id_swaps(tracks, frames)

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True 
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    out_frames = tracker.draw_annotations(frames, tracks, team_ball_control)
    out_frames = camera_movement_estimator.draw_camera_movement(out_frames, camera_movement_per_frame)
    out_frames = speed_and_distance_estimator.draw_speed_and_distance(out_frames, tracks)

    save_video(out_frames, output_path)

def main():
    video_path = 'input_video/08fd33_4.mp4'
    # Read video
    frames = read_video(video_path)
    
    tracker = Tracker('best.pt')

    tracks = tracker.get_obj_trackers(frames, read_from_stub=True, stub_path='stubs/track_stub.pkl')

    tracker.add_position_to_track(tracks)

    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, 
                                                                              read_from_stub=True, 
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    post_processor = PostProcessor(team_assigner)
    tracks = post_processor.merge_fragmented_tracks(tracks)
    tracks = post_processor.fix_id_swaps(tracks, frames)

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True 
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)

    out_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    out_frames = camera_movement_estimator.draw_camera_movement(out_frames, camera_movement_per_frame)

    out_frames = speed_and_distance_estimator.draw_speed_and_distance(out_frames, tracks)

    # Save video
    save_video(out_frames, 'output_video/output.avi')

if __name__ == '__main__':
    main()