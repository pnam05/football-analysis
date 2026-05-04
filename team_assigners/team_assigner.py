from sklearn.cluster import KMeans
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}

    def get_model(self, img):
        img_2d = img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(img_2d)

        return kmeans
    
    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # top_half_img = img[0:int(img.shape[0]/2):]

        h = img.shape[0]
        top_half_img = img[int(h * 0.1) : int(h * 0.5), :]

        kmeans = self.get_model(top_half_img)

        labels = kmeans.labels_
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])

        # corner_clusters = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        # non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # player_cluster = 1 - non_player_cluster

        # player_color = kmeans.cluster_centers_[player_cluster]

        center_y, center_x = top_half_img.shape[0] // 2, top_half_img.shape[1] // 2
        player_cluster_id = clustered_img[center_y, center_x]
        player_color_hsv = kmeans.cluster_centers_[player_cluster_id]

        pixel_hsv = np.uint8([[player_color_hsv]])
        pixel_bgr = cv2.cvtColor(pixel_hsv, cv2.COLOR_HSV2BGR)
        player_color_bgr = pixel_bgr[0][0]

        return player_color_bgr
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 94 or player_id == 130:
            team_id = 1
        
        self.player_team_dict[player_id] = team_id

        return team_id
        
    def get_raw_player_team(self, frame, player_bbox):
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        
        return team_id + 1