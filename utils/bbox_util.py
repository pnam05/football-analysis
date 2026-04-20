def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def get_distance(pts1, pts2):
    return ((pts1[0]-pts2[0])**2 + (pts1[1]-pts2[1])**2)**0.5