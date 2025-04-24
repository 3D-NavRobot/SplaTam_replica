import torch, cv2, kornia as K
import numpy as np
from lightglue import LightGlue, SuperPoint

sp = SuperPoint(max_keypoints=2048).eval().cuda()
lg = LightGlue(features='superpoint').eval().cuda()

@torch.no_grad()
def extract(image_bgr):
    # image_bgr: H×W×3 uint8
    img = torch.tensor(image_bgr, device='cuda').permute(2,0,1).float() / 255.0  # 3×H×W
    gray = K.color.bgr_to_grayscale(img[None])[0]  # 1×H×W
    feats = sp({'image': gray[None]})
    feats['image'] = gray[None]  # so LightGlue can look up image_size if needed
    return feats

def match(f0, f1):
    data = {
      'image0': {'keypoints': f0['keypoints'], 'descriptors': f0['descriptors']},
      'image1': {'keypoints': f1['keypoints'], 'descriptors': f1['descriptors']},
    }
    out = lg(data)
    # matches0: [1×M], batch is always 1 here.
    m = out['matches0'][0]       # now a 1‐D tensor of length M
    return m.cpu().numpy()       # turn into a numpy array

def estimate_pose(kp0, kp1, depth0, intr):
    """
    kp0, kp1 : torch.Tensor [K×2] on CUDA or CPU
    depth0   : torch.Tensor [H×W] or numpy.ndarray [H×W]
    intr     : numpy.ndarray (3×3)
    """
    fx, fy = intr[0,0], intr[1,1]
    cx, cy = intr[0,2], intr[1,2]

    # get integer pixel coords as NumPy arrays
    ys = kp0[:,1].long().cpu().numpy().astype(int)
    xs = kp0[:,0].long().cpu().numpy().astype(int)

    # pull depth into a NumPy array
    if isinstance(depth0, np.ndarray):
        depth_cpu = depth0
    else:
        depth_cpu = depth0.cpu().numpy()

    # index into depth
    z = depth_cpu[ys, xs].astype(np.float32)

    # reproject to 3D
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)
    X = np.vstack([
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z
    ]).T

    # PnP + RANSAC
    succ, rvec, tvec, inliers = cv2.solvePnPRansac(
        X,
        kp1.cpu().numpy().astype(np.float32),
        intr,
        None,
        reprojectionError=3.0,
        iterationsCount=100
    )
    if not succ:
        return None, 0

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3,3] = tvec[:,0]

    return torch.from_numpy(T).cuda(), int(len(inliers))


def track_pair(rgb_ref, depth_ref, rgb_cur, intr):
    """
    rgb_ref, rgb_cur: H×W×3 uint8
    depth_ref: 1×H×W float32
    intr: 3×3 numpy float
    """
    f0 = extract(rgb_ref)
    f1 = extract(rgb_cur)

    m = match(f0, f1)          # numpy array length M of ints in [-1..M-1]
    valid = (m >= 0)
    if valid.sum() < 12:
        return None, 0

    # drop batch dim from keypoints → [M×2]
    kp0_all = f0['keypoints'][0]
    kp1_all = f1['keypoints'][0]

    kp0 = kp0_all[valid]
    kp1 = kp1_all[m[valid]]

    pose, n = estimate_pose(kp0, kp1, depth_ref, intr)

    return pose, n
