import torch, cv2, kornia as K
from lightglue import LightGlue, SuperPoint
import numpy as np

sp = SuperPoint(max_keypoints=2048).eval().cuda()
lg = LightGlue(features='superpoint').eval().cuda()

@torch.no_grad()
def extract(image_bgr):
    # convert to grayscale and normalize
    img = K.color.bgr_to_grayscale(
        torch.tensor(image_bgr).permute(2,0,1)/255.
    ).cuda()
    # run SuperPoint
    feats = sp({'image': img[None]})
    # stash the image so we can pass it to LightGlue
    feats['image'] = img[None]    # shape (1,1,H,W)
    return feats

def match(f0, f1):
    # wrap each SP output into the LightGlue “image” dict
    data0 = {
      "keypoints":   f0["keypoints"][None],       # add batch dim
      "descriptors": f0["descriptors"][None],
      "image":       f0["image"],                 # already [1×1×H×W]
    }
    data1 = {
      "keypoints":   f1["keypoints"][None],
      "descriptors": f1["descriptors"][None],
      "image":       f1["image"],
    }
    out = lg({"image0": data0, "image1": data1})
    # out["matches0"] is [1×M], so drop the batch dim
    return out["matches0"][0].cpu()


def estimate_pose(kp0, kp1, depth0, intr):
    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    z = depth0[kp0[:,1].long(), kp0[:,0].long()]
    X = np.vstack([
      (kp0[:,0]-cx)*z/fx,
      (kp0[:,1]-cy)*z/fy,
      z
    ]).T
    succ, rvec, tvec, inl = cv2.solvePnPRansac(
      X, kp1.float().numpy(), intr, None,
      reprojectionError=3.0, iterationsCount=100
    )
    if not succ:
      return None, 0
    R,_ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec[:,0]
    return torch.tensor(T).float().cuda(), len(inl)

def track_pair(rgb_ref, depth_ref, rgb_cur, intr):
    f0 = extract(rgb_ref)
    f1 = extract(rgb_cur)
    m  = match(f0, f1)
    valid = m > -1
    if valid.sum() < 12:
      return None, 0
    kp0 = f0['keypoints'][valid]
    kp1 = f1['keypoints'][m[valid]]
    pose, n = estimate_pose(kp0, kp1, depth_ref[0], intr)
    return pose, n
