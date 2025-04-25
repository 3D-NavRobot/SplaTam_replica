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
from collections import deque

# Load multiple feature extractors
superpoint = SuperPoint(max_keypoints=2048).eval().cuda()
disk = DISK(max_keypoints=2048).eval().cuda()
aliked = ALIKED(max_keypoints=2048).eval().cuda()
sift = SIFT(max_keypoints=2048).eval().cuda()
doghardnet = DoGHardNet(max_keypoints=2048).eval().cuda()

# Load matchers
lightglue_sp = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).eval().cuda()
lightglue_disk = LightGlue(features='disk', depth_confidence=0.9, width_confidence=0.95).eval().cuda()
lightglue_aliked = LightGlue(features='aliked', depth_confidence=0.9, width_confidence=0.95).eval().cuda()
lightglue_sift = LightGlue(features='sift', depth_confidence=0.9, width_confidence=0.95).eval().cuda()
lightglue_doghardnet = LightGlue(features='doghardnet', depth_confidence=0.9, width_confidence=0.95).eval().cuda()

# Recent history for performance tracking
history_window_size = 10
pose_errors = {}
feature_performance = {
    'superpoint': deque(maxlen=history_window_size),
    'disk': deque(maxlen=history_window_size),
    'aliked': deque(maxlen=history_window_size),
    'sift': deque(maxlen=history_window_size),
    'doghardnet': deque(maxlen=history_window_size)
}

# Ordered list of extractors (will be updated based on performance)
extractor_order = ['superpoint', 'disk', 'aliked', 'sift', 'doghardnet']

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

def update_extractor_order():
    """Update the order of extractors based on their recent performance"""
    global extractor_order
    
    # Calculate average performance for each extractor
    avg_performance = {}
    for name, history in feature_performance.items():
        if len(history) > 0:
            # Average of recent performance metrics (higher is better)
            avg_performance[name] = sum(history) / len(history)
        else:
            avg_performance[name] = 0
    
    # Sort extractors by performance (descending)
    extractor_order = sorted(avg_performance.keys(), key=lambda x: avg_performance[x], reverse=True)

def update_feature_performance(name, performance_metric):
    """Update the performance history for a feature extractor"""
    feature_performance[name].append(performance_metric)
    
    # Update the order of extractors periodically
    if sum(len(history) for history in feature_performance.values()) % 10 == 0:
        update_extractor_order()

def calculate_performance_metric(result):
    """Calculate a performance metric for a tracking result"""
    if result['pose'] is None:
        return 0
    
    # Combine multiple quality indicators
    inlier_count = result['inliers']
    inlier_ratio = result['inlier_ratio']
    
    # Higher values = better performance
    return inlier_count * inlier_ratio

def calculate_pose_error(pose1, pose2):
    """Calculate error between two poses (translation + rotation)"""
    if pose1 is None or pose2 is None:
        return float('inf')
    
    # Translation error
    trans_error = torch.norm(pose1[:3, 3] - pose2[:3, 3])
    
    # Rotation error (using Frobenius norm of difference)
    rot_error = torch.norm(pose1[:3, :3] - pose2[:3, :3])
    
    # Combined error
    return trans_error + 0.1 * rot_error

def track_pair_greedy_soup(rgb_ref, depth_ref, rgb_cur, intr):
    """
    Greedy model soup approach that sequentially adds models if they improve performance
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
    
    # Process each extractor in order of recent performance
    for name in extractor_order:
        extractor, matcher = extractors[name]
        
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
        match_count = valid.sum().item()
        pose, ninliers = None, 0
        
        if match_count >= 12:
            kp0 = data0['keypoints'][valid]
            kp1 = data1['keypoints'][matches[valid]]
            pose, ninliers = estimate_pose(kp0, kp1, depth_ref[0], intr)
        
        # Store result with additional metrics
        inlier_ratio = ninliers / match_count if match_count > 0 else 0
        results[name] = {
            'pose': pose,
            'inliers': ninliers,
            'inlier_ratio': inlier_ratio,
            'match_count': match_count,
            'feature_count': len(data0['keypoints']) if 'keypoints' in data0 else 0
        }
        
        # Update performance metrics
        performance = calculate_performance_metric(results[name])
        update_feature_performance(name, performance)
    
    # Greedy soup approach: Start with the best model and add others if they improve
    best_model_name = max(results.keys(), key=lambda x: calculate_performance_metric(results[x]))
    current_soup = results[best_model_name]['pose']
    current_quality = calculate_performance_metric(results[best_model_name])
    
    if current_soup is None:
        return None, 0  # No valid poses found
    
    # Create a list of models ordered by performance
    model_names_by_performance = sorted(
        [name for name in results.keys() if results[name]['pose'] is not None],
        key=lambda x: calculate_performance_metric(results[x]),
        reverse=True
    )
    
    # Remove the best model as it's already in the soup
    if best_model_name in model_names_by_performance:
        model_names_by_performance.remove(best_model_name)
    
    # Try adding each model to the soup
    soup_models = [best_model_name]
    max_inliers = results[best_model_name]['inliers']
    
    for name in model_names_by_performance:
        if results[name]['pose'] is None:
            continue
            
        # Try adding this model to the soup
        temp_soup = current_soup.clone()
        
        # Simple averaging for now (could be weighted)
        alpha = len(soup_models) / (len(soup_models) + 1)  # Weight for current soup
        temp_soup[:3, :3] = alpha * current_soup[:3, :3] + (1 - alpha) * results[name]['pose'][:3, :3]
        temp_soup[:3, 3] = alpha * current_soup[:3, 3] + (1 - alpha) * results[name]['pose'][:3, 3]
        
        # Orthogonalize the rotation matrix
        u, _, v = torch.svd(temp_soup[:3, :3])
        temp_soup[:3, :3] = u @ v.T
        
        # Test the new soup on a proxy metric
        # In a real implementation, you'd test on a validation set
        # Here we use inlier count as a proxy for accuracy
        proxy_quality = 0
        for test_name in soup_models + [name]:
            if results[test_name]['pose'] is not None:
                error = calculate_pose_error(temp_soup, results[test_name]['pose'])
                proxy_quality += results[test_name]['inliers'] / (1 + error)
        
        # Only add to soup if it improves quality
        if proxy_quality > current_quality:
            current_soup = temp_soup
            current_quality = proxy_quality
            soup_models.append(name)
            max_inliers = max(max_inliers, results[name]['inliers'])
    
    # Return the final soup
    return current_soup, max_inliers

def track_pair_simple_best(rgb_ref, depth_ref, rgb_cur, intr):
    """
    Simple approach that just uses the best performing extractor
    rather than a soup. Useful for comparison.
    """
    # Process each extractor in order of recent performance
    for name in extractor_order[:2]:  # Only try the top 2 extractors
        extractor, matcher = {
            'superpoint': (superpoint, lightglue_sp),
            'disk': (disk, lightglue_disk),
            'aliked': (aliked, lightglue_aliked),
            'sift': (sift, lightglue_sift),
            'doghardnet': (doghardnet, lightglue_doghardnet)
        }[name]
        
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
        match_count = valid.sum().item()
        pose, ninliers = None, 0
        
        if match_count >= 12:
            kp0 = data0['keypoints'][valid]
            kp1 = data1['keypoints'][matches[valid]]
            pose, ninliers = estimate_pose(kp0, kp1, depth_ref[0], intr)
            
            # Update performance metrics
            inlier_ratio = ninliers / match_count if match_count > 0 else 0
            performance = ninliers * inlier_ratio
            update_feature_performance(name, performance)
            
            # If we got a good pose, return it
            if pose is not None and ninliers > 20:  # Minimum quality threshold
                return pose, ninliers
    
    # If no good pose found, return None
    return None, 0

# Usage in SplaTAM's rgbd_slam function:
# Replace the original track_pair call with:
# pose_feat, ninl = track_pair_greedy_soup(
#     prev_rgb_bgr,
#     prev_depth_cpu,
#     (color.permute(1,2,0)*255).byte().cpu().numpy(),
#     intrinsics.cpu().numpy()
# )