from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        detections = []
        batch_size = 20

        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size], conf=0.4)
            detections += detection

        return detections
    
    def get_obj_trackers(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            for idx, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_sv.class_id[idx] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for detection in detection_with_tracks:
                bbox = detection[0].tolist()
                cls_id = detection[3]
                track_id = detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for detection in detection_sv:
                bbox = detection[0].tolist()
                cls_id = detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = bbox
        x_center, _ = (int((x1+x2)/2), int((y1+y2)/2))
        width = x2 - x1

        cv2.ellipse(
            frame,
            center=(x_center, int(y2)),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            if track_id < 10:
                x1_text = x1_rect + 12
            if track_id > 9 and track_id < 100:
                x1_text = x1_rect + 7
            elif track_id > 100:
                x1_text = x1_rect + 2
            
            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                thickness=2
            )
        return frame 
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x_center, _ = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))

        triangle_pts = np.asarray([
            [x_center, y],
            [x_center-10, y-20],
            [x_center+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_pts], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        h, w, _ = frame.shape

        x1, y1 = int(0.70 * w), int(0.80 * h)
        x2, y2 = int(0.98 * w), int(0.90 * h)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        team_1_frames = (team_ball_control_till_frame == 1).sum()
        team_2_frames = (team_ball_control_till_frame == 2).sum()

        total = team_1_frames + team_2_frames
        if total == 0:
            team_1 = team_2 = 0
        else:
            team_1 = team_1_frames / total
            team_2 = 1 - team_1

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",
                    (x1+35, y1+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",
                    (x1+35, y1+85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        return frame


    def draw_annotations(self, frames, tracks, team_ball_control):
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (255, 0, 0))



            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
            
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output.append(frame)
        
        return output


                



