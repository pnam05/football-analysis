import numpy as np
import sys
sys.path.append('../')
from utils import get_center

class PostProcessor:
    def __init__(self, team_assigner):
        self.team_assigner = team_assigner
        self.distance_threshold = 80  
        self.frame_buffer_past = 30  
        self.frame_buffer_future = 30 

    def fix_id_swaps(self, tracks, frames):
        players_tracks = tracks['players']
        swap_count = {}
        
        # --- THÊM MỚI: Dictionary để khóa các cặp ID vừa được sửa ---
        cooldowns = {} 

        for frame_num in range(self.frame_buffer_past, len(players_tracks) - self.frame_buffer_future):
            current_frame_tracks = players_tracks[frame_num]
            player_ids = list(current_frame_tracks.keys())

            swapped_this_frame = False
            for i in range(len(player_ids)):
                if swapped_this_frame:
                    break
                for j in range(i + 1, len(player_ids)):
                    id1, id2 = player_ids[i], player_ids[j]
                    pair_key = (min(id1, id2), max(id1, id2))

                    # --- THÊM MỚI: Bỏ qua nếu cặp này đang trong thời gian Cooldown ---
                    if frame_num < cooldowns.get(pair_key, 0):
                        continue

                    c1 = np.array(get_center(current_frame_tracks[id1]['bbox']))
                    c2 = np.array(get_center(current_frame_tracks[id2]['bbox']))
                    if np.sum((c1 - c2) ** 2) >= self.distance_threshold ** 2:
                        continue

                    past_frame = frame_num - self.frame_buffer_past
                    future_frame = frame_num + self.frame_buffer_future

                    past_team1 = self.get_team_majority(frames, players_tracks, id1, past_frame)
                    past_team2 = self.get_team_majority(frames, players_tracks, id2, past_frame)
                    future_team1 = self.get_team_majority(frames, players_tracks, id1, future_frame)
                    future_team2 = self.get_team_majority(frames, players_tracks, id2, future_frame)

                    if None in (past_team1, past_team2, future_team1, future_team2):
                        continue
                    if past_team1 == past_team2:
                        continue

                    if future_team1 == past_team2 and future_team2 == past_team1:
                        print(f"-> [Fixed] ID Swap giữa {id1} và {id2} tại frame {frame_num}")
                        swap_count[pair_key] = swap_count.get(pair_key, 0) + 1

                        # --- THÊM MỚI: Cập nhật thời gian Cooldown ---
                        # Công thức: frame hiện tại + past + future + một khoảng đệm an toàn (ví dụ 5 frames)
                        # Đảm bảo past_frame của lần check kế tiếp sẽ nằm hoàn toàn bên ngoài vùng tranh chấp hiện tại.
                        cooldown_period = self.frame_buffer_past + self.frame_buffer_future + 5
                        cooldowns[pair_key] = frame_num + cooldown_period

                        for f in range(frame_num, len(players_tracks)):
                            track_f = players_tracks[f]
                            has_1, has_2 = id1 in track_f, id2 in track_f
                            temp1 = track_f.pop(id1, None)
                            temp2 = track_f.pop(id2, None)
                            if has_2: track_f[id1] = temp2
                            if has_1: track_f[id2] = temp1

                        swapped_this_frame = True
                        break

        return tracks

    def get_team_majority(self, frames, players_tracks, track_id, center_frame, window=3):
        votes = []
        for offset in range(-window, window + 1):
            f = center_frame + offset
            if 0 <= f < len(players_tracks) and track_id in players_tracks[f]:
                bbox = players_tracks[f][track_id]['bbox']
                team = self.team_assigner.get_raw_player_team(frames[f], bbox)
                if team is not None:
                    votes.append(team)
        if not votes:
            return None
        return max(set(votes), key=votes.count)
    
    def merge_fragmented_tracks(self, tracks, max_lost_frames=60, max_distance=150):
        players_tracks = tracks['players']
        
        while True:
            merged_in_this_pass = False
            
            # Bước 1: Thu thập vòng đời của từng ID ở mỗi pass
            track_lifespans = {}
            for frame_num, frame_tracks in enumerate(players_tracks):
                for track_id, info in frame_tracks.items():
                    if track_id not in track_lifespans:
                        track_lifespans[track_id] = {
                            'start_frame': frame_num,
                            'end_frame': frame_num,
                            'first_bbox': info['bbox'],
                            'last_bbox': info['bbox']
                        }
                    else:
                        track_lifespans[track_id]['end_frame'] = frame_num
                        track_lifespans[track_id]['last_bbox'] = info['bbox']

            # Bước 2: Tìm các cặp ID bị đứt gãy để nối lại
            for old_id, old_info in track_lifespans.items():
                for new_id, new_info in track_lifespans.items():
                    if old_id == new_id:
                        continue
                    
                    frame_gap = new_info['start_frame'] - old_info['end_frame']
                    
                    # Nếu B xuất hiện sau khi A biến mất (trong khoảng max_lost_frames)
                    if 0 < frame_gap <= max_lost_frames:
                        
                        # Kiểm tra khoảng cách vật lý
                        c_old = np.array(get_center(old_info['last_bbox']))
                        c_new = np.array(get_center(new_info['first_bbox']))
                        distance = np.linalg.norm(c_old - c_new)

                        if distance <= max_distance:
                            print(f"-> [Stitched] Nối ID {old_id} với ID mới {new_id} (Gap: {frame_gap} frames)")
                            
                            # Đổi toàn bộ new_id thành old_id trong dữ liệu track
                            for f in range(new_info['start_frame'], new_info['end_frame'] + 1):
                                if new_id in players_tracks[f]:
                                    players_tracks[f][old_id] = players_tracks[f].pop(new_id)
                            
                            merged_in_this_pass = True
                            break # Phá vòng lặp new_id
                            
                if merged_in_this_pass:
                    break # Phá vòng lặp old_id để quét lại từ đầu với dữ liệu đã được update

            # Nếu quét qua toàn bộ mà không có ID nào được nối thêm, thoát vòng lặp while
            if not merged_in_this_pass:
                break

        return tracks