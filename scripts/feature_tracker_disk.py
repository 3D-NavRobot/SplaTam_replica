"""
Fast feature‑based tracking for SplaTAM
--------------------------------------
• Extract SuperPoint key‑points & descriptors (GPU).
• Match consecutive frames with LightGlue (GPU).
• Estimate relative SE(3) via OpenCV RANSAC‑PnP.
• Return 4×4 w2c + inlier count (quality metric).
"""

import torch, cv2, kornia as K
from lightglue import LightGlue, SuperPoint, DISK
import numpy as np

disk = DISK(max_num_keypoints=2048).eval().cuda() 
lg = LightGlue(features='disk').eval().cuda()

@torch.no_grad()
def extract(image_bgr):
    img = K.color.bgr_to_grayscale(torch.tensor(image_bgr).permute(2,0,1)/255.).cuda()
    return disk({'image': img[None]})

def match(desc0, desc1, kpts0, kpts1):
    data = {'descriptors0': desc0, 'descriptors1': desc1,
            'keypoints0': kpts0, 'keypoints1': kpts1}
    return lg(data)['matches0'].cpu()

def estimate_pose(kp0, kp1, depth0, intr):
    # back‑project reference points
    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    z = depth0[kp0[:,1].long(), kp0[:,0].long()]
    X = np.vstack([(kp0[:,0]-cx)*z/fx, (kp0[:,1]-cy)*z/fy, z]).T
    # PnP RANSAC
    succ, rvec, tvec, inl = cv2.solvePnPRansac(
        X, kp1.float().numpy(), intr, None, reprojectionError=3.0, iterationsCount=100)
    if not succ: return None, 0
    R,_ = cv2.Rodrigues(rvec); T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec[:,0]
    return torch.tensor(T).float().cuda(), len(inl)

def track_pair(rgb_ref, depth_ref, rgb_cur, intr):
    d0 = extract(rgb_ref); d1 = extract(rgb_cur)
    m = match(d0['descriptors'], d1['descriptors'],
              d0['keypoints'],  d1['keypoints'])
    valid = m > -1
    if valid.sum() < 12: return None, 0
    kp0 = d0['keypoints'][valid]; kp1 = d1['keypoints'][m[valid]]
    pose, n = estimate_pose(kp0, kp1, depth_ref[0], intr)
    return pose, n
