from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        detections = []
        batch_size = 20

        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size], conf=0.1)
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
            x1_text = x1_rect + 12
            if track_id > 9:
                x1_text -= 10
            
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

    def draw_annatations(self, frames, tracks):
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            
            output.append(frame)
        
        return output


                



