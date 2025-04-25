import torch
import cv2
import kornia as K
import numpy as np
from lightglue import LightGlue
from lightglue.superpoint import SuperPoint
from lightglue.disk import DISK
from lightglue.aliked import ALIKED
from lightglue.sift import SIFT
from lightglue.dog_hardnet import DoGHardNet

# Load multiple feature extractors
superpoint = SuperPoint(max_keypoints=2048).eval().cuda()
disk = DISK(max_keypoints=2048).eval().cuda()
aliked = ALIKED(max_keypoints=2048).eval().cuda()
sift = SIFT(max_keypoints=2048).eval().cuda()
doghardnet = DoGHardNet(max_keypoints=2048).eval().cuda()

# Load matchers
lightglue_sp = LightGlue(features='superpoint').eval().cuda()
lightglue_disk = LightGlue(features='disk').eval().cuda()
lightglue_aliked = LightGlue(features='aliked').eval().cuda()
lightglue_sift = LightGlue(features='sift').eval().cuda()
lightglue_doghardnet = LightGlue(features='doghardnet').eval().cuda()

@torch.no_grad()
def extract_features(image_bgr, extractor):
    """Generic feature extraction function for all extractors"""
    img = K.color.bgr_to_grayscale(torch.tensor(image_bgr).permute(2,0,1)/255.).cuda()
    return extractor({'image': img[None]})

def match_features(desc0, desc1, kpts0, kpts1, matcher):
    """Generic feature matching function for all matchers"""
    data = {'descriptors0': desc0, 'descriptors1': desc1,
            'keypoints0': kpts0, 'keypoints1': kpts1}
    return matcher(data)['matches0'].cpu()

def estimate_pose(kp0, kp1, depth0, intr, ransac_threshold=3.0):
    # back‑project reference points
    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
    z = depth0[kp0[:,1].long(), kp0[:,0].long()]
    X = np.vstack([(kp0[:,0]-cx)*z/fx, (kp0[:,1]-cy)*z/fy, z]).T
    
    # PnP RANSAC
    succ, rvec, tvec, inl = cv2.solvePnPRansac(
        X, kp1.float().numpy(), intr, None, 
        reprojectionError=ransac_threshold, 
        iterationsCount=100)
    
    if not succ: 
        return None, 0
    
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = tvec[:,0]
    
    return torch.tensor(T).float().cuda(), len(inl)

def track_pair_soup(rgb_ref, depth_ref, rgb_cur, intr):
    """
    Model soup approach that combines multiple feature extractors and matchers
    """
    # Dictionary to store results for each extractor
    results = {}
    extractors = {
        'superpoint': (superpoint, lightglue_sp),
        'disk': (disk, lightglue_disk),
        'aliked': (aliked, lightglue_aliked),
        'sift': (sift, lightglue_sift),
        'doghardnet': (doghardnet, lightglue_doghardnet)
    }
    
    # Process each extractor
    for name, (extractor, matcher) in extractors.items():
        # Extract features
        data0 = extract_features(rgb_ref, extractor)
        data1 = extract_features(rgb_cur, extractor)
        
        # Match features
        matches = match_features(
            data0['descriptors'], data1['descriptors'],
            data0['keypoints'], data1['keypoints'],
            matcher
        )
        
        # Estimate pose
        valid = matches > -1
        pose, ninliers = None, 0
        
        if valid.sum() >= 12:
            kp0 = data0['keypoints'][valid]
            kp1 = data1['keypoints'][matches[valid]]
            pose, ninliers = estimate_pose(kp0, kp1, depth_ref[0], intr)
        
        results[name] = {'pose': pose, 'inliers': ninliers}
    
    # Model soup decision logic
    best_pose = None
    max_inliers = 0
    poses = []
    weights = []
    
    # Collect valid poses with their weights (based on inlier count)
    for name, result in results.items():
        if result['pose'] is not None:
            poses.append(result['pose'])
            weights.append(result['inliers'])
            
            if result['inliers'] > max_inliers:
                best_pose = result['pose']
                max_inliers = result['inliers']
    
    # If multiple valid poses, create a weighted average (model soup)
    if len(poses) > 1:
        # Convert weights to probabilities
        weights = np.array(weights) / sum(weights)
        
        # For rotation, convert to quaternion for weighted average
        avg_translation = torch.zeros(3, device=poses[0].device)
        avg_rotation = torch.zeros((3, 3), device=poses[0].device)
        
        for i, pose in enumerate(poses):
            avg_translation += weights[i] * pose[:3, 3]
            avg_rotation += weights[i] * pose[:3, :3]
        
        # Orthogonalize the rotation matrix (Gram-Schmidt)
        u, _, v = torch.svd(avg_rotation)
        avg_rotation = u @ v.T
        
        # Create the final pose
        final_pose = torch.eye(4, device=poses[0].device)
        final_pose[:3, :3] = avg_rotation
        final_pose[:3, 3] = avg_translation
        
        return final_pose, max_inliers
    
    # If only one valid pose or no valid poses
    return best_pose, max_inliers

# Additional function implementing model soup with adaptive weights
def track_pair_soup_adaptive(rgb_ref, depth_ref, rgb_cur, intr):
    """
    Advanced model soup approach with adaptive weighting based on scene characteristics
    """
    # First, get all the poses using the regular model soup
    results = {}
    extractors = {
        'superpoint': (superpoint, lightglue_sp),
        'disk': (disk, lightglue_disk),
        'aliked': (aliked, lightglue_aliked),
        'sift': (sift, lightglue_sift),
        'doghardnet': (doghardnet, lightglue_doghardnet)
    }
    
    # Process each extractor
    for name, (extractor, matcher) in extractors.items():
        # Extract features
        data0 = extract_features(rgb_ref, extractor)
        data1 = extract_features(rgb_cur, extractor)
        
        # Match features
        matches = match_features(
            data0['descriptors'], data1['descriptors'],
            data0['keypoints'], data1['keypoints'],
            matcher
        )
        
        # Estimate pose
        valid = matches > -1
        pose, ninliers = None, 0
        
        if valid.sum() >= 12:
            kp0 = data0['keypoints'][valid]
            kp1 = data1['keypoints'][matches[valid]]
            pose, ninliers = estimate_pose(kp0, kp1, depth_ref[0], intr)
            
            # Store additional info for adaptive weighting
            results[name] = {
                'pose': pose, 
                'inliers': ninliers,
                'inlier_ratio': ninliers / valid.sum() if valid.sum() > 0 else 0,
                'feature_count': len(data0['keypoints']),
                'match_count': valid.sum().item()
            }
        else:
            results[name] = {
                'pose': None, 
                'inliers': 0,
                'inlier_ratio': 0,
                'feature_count': len(data0['keypoints']) if 'keypoints' in data0 else 0,
                'match_count': 0
            }
    
    # Collect valid poses
    poses = []
    # More sophisticated weighting scheme 
    weights = []
    
    for name, result in results.items():
        if result['pose'] is not None:
            poses.append(result['pose'])
            
            # Weight = inlier_count * inlier_ratio
            # This gives higher weight to methods with both many inliers and high precision
            weight = result['inliers'] * result['inlier_ratio']
            weights.append(weight)
    
    # If no valid poses, return None
    if len(poses) == 0:
        return None, 0
    
    # If only one valid pose, return it
    if len(poses) == 1:
        return poses[0], results[list(results.keys())[0]]['inliers']
    
    # For multiple valid poses, create a weighted average
    weights = np.array(weights) / sum(weights)
    
    # For rotation, convert to quaternion for weighted average
    avg_translation = torch.zeros(3, device=poses[0].device)
    avg_rotation = torch.zeros((3, 3), device=poses[0].device)
    
    for i, pose in enumerate(poses):
        avg_translation += weights[i] * pose[:3, 3]
        avg_rotation += weights[i] * pose[:3, :3]
    
    # Orthogonalize the rotation matrix (Gram-Schmidt)
    u, _, v = torch.svd(avg_rotation)
    avg_rotation = u @ v.T
    
    # Create the final pose
    final_pose = torch.eye(4, device=poses[0].device)
    final_pose[:3, :3] = avg_rotation
    final_pose[:3, 3] = avg_translation
    
    # Return the max inliers as a quality metric
    max_inliers = max([r['inliers'] for r in results.values()])
    
    return final_pose, max_inliers

# Usage in SplaTAM's rgbd_slam function:
# Replace the original track_pair call with:
# pose_feat, ninl = track_pair_soup(
#     prev_rgb_bgr,
#     prev_depth_cpu,
#     (color.permute(1,2,0)*255).byte().cpu().numpy(),
#     intrinsics.cpu().numpy()
# )
