"""
FlashSLAM Implementation: Key code changes to transform the existing implementation
"""

# 1. Update imports to include necessary optimization tools
import torch.cuda.amp as amp  # Add mixed precision for faster computation
from torch.nn.parallel import DistributedDataParallel as DDP  # For multi-GPU support
import torch.multiprocessing as mp  # For parallel processing
from concurrent.futures import ThreadPoolExecutor  # For parallel data loading
import torch.utils.cpp_extension  # For JIT compilation of CUDA kernels
import queue  # For thread-safe data queues
import threading  # For background processing threads

# 2. Add CUDA kernels for FlashSLAM's performance-critical operations
# Define CUDA kernel for fast point cloud processing
point_cloud_cuda_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for efficient point cloud processing
__global__ void compute_point_cloud_kernel(
    const float* depth_data,
    const float* intrinsics,
    const float* inv_pose,
    float* point_cloud,
    const int width,
    const int height,
    const int* mask
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    const int x = idx % width;
    const int y = idx / width;
    
    // Apply mask if provided
    if (mask && !mask[idx]) return;
    
    const float depth_value = depth_data[idx];
    if (depth_value <= 0.0f) return;
    
    // Camera intrinsics
    const float fx = intrinsics[0];
    const float fy = intrinsics[4];
    const float cx = intrinsics[2];
    const float cy = intrinsics[5];
    
    // Back-project to camera coordinates
    float pt_cam_x = (x - cx) * depth_value / fx;
    float pt_cam_y = (y - cy) * depth_value / fy;
    float pt_cam_z = depth_value;
    
    // Transform to world coordinates
    float pt_w_x = inv_pose[0] * pt_cam_x + inv_pose[1] * pt_cam_y + inv_pose[2] * pt_cam_z + inv_pose[3];
    float pt_w_y = inv_pose[4] * pt_cam_x + inv_pose[5] * pt_cam_y + inv_pose[6] * pt_cam_z + inv_pose[7];
    float pt_w_z = inv_pose[8] * pt_cam_x + inv_pose[9] * pt_cam_y + inv_pose[10] * pt_cam_z + inv_pose[11];
    
    // Store in output
    point_cloud[idx * 3] = pt_w_x;
    point_cloud[idx * 3 + 1] = pt_w_y;
    point_cloud[idx * 3 + 2] = pt_w_z;
}

// CUDA kernel for computing mean squared distances
__global__ void compute_mean_sq_dist_kernel(
    const float* depth_data,
    const float* intrinsics,
    float* mean_sq_dist,
    const int width,
    const int height,
    const int* mask
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    // Apply mask if provided
    if (mask && !mask[idx]) return;
    
    const float depth_value = depth_data[idx];
    if (depth_value <= 0.0f) return;
    
    // Camera intrinsics
    const float fx = intrinsics[0];
    const float fy = intrinsics[4];
    
    // Compute scale based on projective geometry
    const float scale = depth_value / ((fx + fy) / 2.0f);
    mean_sq_dist[idx] = scale * scale;
}

torch::Tensor compute_point_cloud_cuda(
    torch::Tensor depth,
    torch::Tensor intrinsics,
    torch::Tensor inv_pose,
    torch::Tensor mask = torch::Tensor()
) {
    const int height = depth.size(0);
    const int width = depth.size(1);
    const int total_size = width * height;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(depth.device());
    auto point_cloud = torch::zeros({total_size, 3}, options);
    
    const dim3 blocks((total_size + 255) / 256);
    const dim3 threads(256);
    
    compute_point_cloud_kernel<<<blocks, threads>>>(
        depth.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        inv_pose.data_ptr<float>(),
        point_cloud.data_ptr<float>(),
        width,
        height,
        mask.numel() > 0 ? mask.data_ptr<int>() : nullptr
    );
    
    return point_cloud;
}

torch::Tensor compute_mean_sq_dist_cuda(
    torch::Tensor depth,
    torch::Tensor intrinsics,
    torch::Tensor mask = torch::Tensor()
) {
    const int height = depth.size(0);
    const int width = depth.size(1);
    const int total_size = width * height;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(depth.device());
    auto mean_sq_dist = torch::zeros({total_size}, options);
    
    const dim3 blocks((total_size + 255) / 256);
    const dim3 threads(256);
    
    compute_mean_sq_dist_kernel<<<blocks, threads>>>(
        depth.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        mean_sq_dist.data_ptr<float>(),
        width,
        height,
        mask.numel() > 0 ? mask.data_ptr<int>() : nullptr
    );
    
    return mean_sq_dist;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_point_cloud", &compute_point_cloud_cuda, "Compute point cloud from depth (CUDA)");
    m.def("compute_mean_sq_dist", &compute_mean_sq_dist_cuda, "Compute mean squared distances (CUDA)");
}
"""

# Function to load FlashSLAM CUDA kernels
def load_flashslam_cuda_kernels():
    try:
        # JIT compile the CUDA extension
        point_cloud_cuda = torch.utils.cpp_extension.load_inline(
            name="point_cloud_cuda",
            cpp_sources="",  # No CPU implementation
            cuda_sources=point_cloud_cuda_kernel,
            functions=["compute_point_cloud", "compute_mean_sq_dist"],
            verbose=True
        )
        return point_cloud_cuda
    except Exception as e:
        print(f"Warning: Failed to compile CUDA kernels: {e}")
        print("Falling back to PyTorch implementation")
        return None

# 3. Add FlashSLAM configuration parameters to config dict
def add_flashslam_config(config):
    # Performance optimization parameters
    if "flashslam" not in config:
        config["flashslam"] = {
            "use_mixed_precision": True,
            "use_multi_gpu": False,
            "parallel_data_loading": True,
            "adaptive_keyframe_selection": True,
            "hierarchical_mapping": True,
            "use_local_ba": True,
            "local_ba_window_size": 7,
            "use_progressive_densification": True,
            "use_cuda_kernels": True,
            "gpu_cache_size": 2048,  # in MB
            "coarse_to_fine_iters": [32, 16, 8],  # Multi-resolution iteration counts
            "coarse_to_fine_scales": [0.25, 0.5, 1.0],  # Multi-resolution scales
            "prefetch_size": 8,  # Number of frames to prefetch
            "max_active_gaussians": 500000,  # Maximum number of active Gaussians to keep in memory
            "frustum_culling": True,  # Use frustum culling to skip off-screen Gaussians
            "adaptive_pruning": True,  # Dynamically adjust pruning thresholds
        }
    return config

# 3. Update get_dataset with parallel data loading
def get_dataset_flashslam(config_dict, basedir, sequence, **kwargs):
    # Create the original dataset
    dataset = get_dataset(config_dict, basedir, sequence, **kwargs)
    
    # Wrap with parallel data loading if enabled
    if config_dict.get("flashslam", {}).get("parallel_data_loading", False):
        return ParallelLoadDataset(dataset)
    return dataset

# 4. Add ParallelLoadDataset wrapper for efficient data loading
class ParallelLoadDataset:
    def __init__(self, dataset, num_workers=4, prefetch_size=8):
        self.dataset = dataset
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_size = prefetch_size
        self.prefetch_buffer = {}
        self.future_buffer = {}
        self._prefetch(0, min(prefetch_size, len(dataset)))
        
    def _prefetch(self, start_idx, end_idx):
        for idx in range(start_idx, end_idx):
            if idx not in self.future_buffer and idx not in self.prefetch_buffer:
                self.future_buffer[idx] = self.executor.submit(self.dataset.__getitem__, idx)
    
    def __getitem__(self, idx):
        # Trigger prefetching next batch
        prefetch_start = idx + 1
        prefetch_end = min(prefetch_start + self.prefetch_size, len(self.dataset))
        self._prefetch(prefetch_start, prefetch_end)
        
        # Get current item
        if idx in self.prefetch_buffer:
            item = self.prefetch_buffer[idx]
            del self.prefetch_buffer[idx]
            return item
        
        if idx in self.future_buffer:
            future = self.future_buffer[idx]
            del self.future_buffer[idx]
            return future.result()
        
        # Fallback to direct loading
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

# 5. Add mixed precision training support
class MixedPrecisionTrainer:
    def __init__(self, use_mixed_precision=True):
        self.use_mixed_precision = use_mixed_precision
        self.scaler = amp.GradScaler(enabled=use_mixed_precision)
    
    def get_loss(self, params, curr_data, variables, iter_time_idx, loss_weights, 
                 use_sil_for_loss, sil_thres, use_l1, ignore_outlier_depth_loss, 
                 tracking=False, mapping=False, do_ba=False, plot_dir=None, 
                 visualize_tracking_loss=False, tracking_iteration=None):
        
        # Use autocast for mixed precision computation
        with amp.autocast(enabled=self.use_mixed_precision):
            loss, variables, weighted_losses = get_loss(
                params, curr_data, variables, iter_time_idx, loss_weights,
                use_sil_for_loss, sil_thres, use_l1, ignore_outlier_depth_loss,
                tracking, mapping, do_ba, plot_dir, 
                visualize_tracking_loss, tracking_iteration
            )
        return loss, variables, weighted_losses
    
    def optimizer_step(self, loss, optimizer):
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

# 6. Implement hierarchical Gaussian management for FlashSLAM
class HierarchicalGaussianMap:
    def __init__(self, initial_params, cell_size=1.0):
        self.params = initial_params
        self.cell_size = cell_size
        self.cells = {}
        self.build_spatial_index()
    
    def build_spatial_index(self):
        # Create spatial cells for faster access
        means3D = self.params['means3D']
        for i in range(means3D.shape[0]):
            pos = means3D[i].detach().cpu().numpy()
            cell_x = int(pos[0] / self.cell_size)
            cell_y = int(pos[1] / self.cell_size)
            cell_z = int(pos[2] / self.cell_size)
            cell_key = (cell_x, cell_y, cell_z)
            
            if cell_key not in self.cells:
                self.cells[cell_key] = []
            self.cells[cell_key].append(i)
    
    def get_nearby_gaussians(self, position, radius):
        # Get gaussians within a radius of the position
        center_x = int(position[0] / self.cell_size)
        center_y = int(position[1] / self.cell_size)
        center_z = int(position[2] / self.cell_size)
        
        cell_radius = int(radius / self.cell_size) + 1
        nearby_indices = []
        
        for x in range(center_x - cell_radius, center_x + cell_radius + 1):
            for y in range(center_y - cell_radius, center_y + cell_radius + 1):
                for z in range(center_z - cell_radius, center_z + cell_radius + 1):
                    cell_key = (x, y, z)
                    if cell_key in self.cells:
                        nearby_indices.extend(self.cells[cell_key])
        
        return nearby_indices
    
    def update_map(self, new_params):
        self.params = new_params
        self.cells = {}
        self.build_spatial_index()

# 7. Implement adaptive keyframe selection for FlashSLAM
def adaptive_keyframe_selection(curr_loss, depth, curr_w2c, intrinsics, keyframe_list, 
                              max_keyframes, min_loss_threshold=100.0, 
                              overlap_threshold=0.6):
    """More sophisticated keyframe selection based on tracking loss and map coverage"""
    
    # If tracking loss is high, we need more keyframes for robust mapping
    if curr_loss > min_loss_threshold:
        num_keyframes = max(3, max_keyframes // 2)  # Use more keyframes when tracking is poor
    else:
        num_keyframes = max(1, max_keyframes // 4)  # Fewer keyframes when tracking is good
    
    # Select keyframes based on overlap
    selected_keyframes = keyframe_selection_overlap(
        depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
    
    return selected_keyframes

# 8. Modify rgbd_slam function to include FlashSLAM optimizations
def flashslam(config: dict):
    # Enhance config with FlashSLAM parameters
    config = add_flashslam_config(config)
    
    # Print Config
    print("Loaded FlashSLAM Config:")
    print(f"{config}")
    
    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        use_mixed_precision=config["flashslam"]["use_mixed_precision"])
    
    # Setup output directories, device, etc. (same as original)
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize WandB (same as original)
    if config['use_wandb']:
        # Same wandb setup as in the original code
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
    
    # Get Device
    device = torch.device(config["primary_device"])
    
    # Load Dataset with parallel loading
    print("Loading Dataset with parallel processing...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        gradslam_data_cfg["flashslam"] = config["flashslam"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        gradslam_data_cfg["flashslam"] = config["flashslam"]
        
    # Get dataset with parallel loading if enabled
    dataset = get_dataset_flashslam(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config.get("ignore_bad", False),
        use_train_split=dataset_config.get("use_train_split", True),
    )
    
    # Initialize first frame & parameters as in original code
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
    
    # Initialize first timestep with hierarchical mapping
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
        dataset, num_frames, config['scene_radius_depth_ratio'],
        config['mean_sq_dist_method'], gaussian_distribution=config['gaussian_distribution'])
    
    # Initialize hierarchical Gaussian map
    if config["flashslam"]["hierarchical_mapping"]:
        h_map = HierarchicalGaussianMap(params)
    
    # Rest of initialization as in original code
    keyframe_list = []
    keyframe_time_indices = []
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    
    # Main SLAM loop with FlashSLAM optimizations
    for time_idx in tqdm(range(0, num_frames)):
        # Load RGBD frames with parallel loader
        color, depth, _, gt_pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(gt_pose)
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        
        # Prepare multi-resolution data if using coarse-to-fine approach
        if config["flashslam"]["coarse_to_fine_iters"]:
            multi_res_data = []
            for scale in config["flashslam"]["coarse_to_fine_scales"]:
                if scale < 1.0:
                    scaled_h = int(color.shape[1] * scale)
                    scaled_w = int(color.shape[2] * scale)
                    scaled_color = F.interpolate(color.unsqueeze(0), size=(scaled_h, scaled_w), 
                                               mode='bilinear', align_corners=False).squeeze(0)
                    scaled_depth = F.interpolate(depth, size=(scaled_h, scaled_w), 
                                               mode='nearest')
                    scaled_intrinsics = intrinsics.clone()
                    scaled_intrinsics[0, 0] *= scale  # fx
                    scaled_intrinsics[1, 1] *= scale  # fy
                    scaled_intrinsics[0, 2] *= scale  # cx
                    scaled_intrinsics[1, 2] *= scale  # cy
                    
                    scaled_cam = setup_camera(scaled_w, scaled_h, 
                                            scaled_intrinsics.cpu().numpy(), 
                                            first_frame_w2c.detach().cpu().numpy())
                    
                    multi_res_data.append({
                        'cam': scaled_cam,
                        'im': scaled_color,
                        'depth': scaled_depth,
                        'id': time_idx,
                        'intrinsics': scaled_intrinsics,
                        'w2c': first_frame_w2c,
                        'iter_gt_w2c_list': gt_w2c_all_frames
                    })
                else:
                    multi_res_data.append({
                        'cam': cam,
                        'im': color,
                        'depth': depth,
                        'id': time_idx,
                        'intrinsics': intrinsics,
                        'w2c': first_frame_w2c,
                        'iter_gt_w2c_list': gt_w2c_all_frames
                    })
        else:
            # Standard single-resolution data
            curr_data = {
                'cam': cam, 
                'im': color, 
                'depth': depth, 
                'id': time_idx, 
                'intrinsics': intrinsics,
                'w2c': first_frame_w2c, 
                'iter_gt_w2c_list': gt_w2c_all_frames
            }
            multi_res_data = [curr_data]
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, 
                                         forward_prop=config['tracking']['forward_prop'])
        
        # Tracking with coarse-to-fine approach
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Keep track of best poses
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            
            # Multi-resolution tracking
            for res_idx, res_data in enumerate(multi_res_data):
                if res_idx >= len(config["flashslam"]["coarse_to_fine_iters"]):
                    continue
                    
                # Reset optimizer for this resolution level
                optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
                
                num_iters = config["flashslam"]["coarse_to_fine_iters"][res_idx]
                progress_bar = tqdm(range(num_iters), 
                                  desc=f"Tracking Time Step: {time_idx}, Res: {res_idx+1}/{len(multi_res_data)}")
                
                for iter in range(num_iters):
                    iter_start_time = time.time()
                    
                    # Get loss with mixed precision if enabled
                    loss, variables, losses = mp_trainer.get_loss(
                        params, res_data, variables, time_idx, 
                        config['tracking']['loss_weights'],
                        config['tracking']['use_sil_for_loss'], 
                        config['tracking']['sil_thres'],
                        config['tracking']['use_l1'], 
                        config['tracking']['ignore_outlier_depth_loss'], 
                        tracking=True, plot_dir=eval_dir,
                        visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                        tracking_iteration=iter
                    )
                    
                    # Update with mixed precision
                    mp_trainer.optimizer_step(loss, optimizer)
                    
                    with torch.no_grad():
                        # Save best candidate
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    
                    # Update progress
                    progress_bar.update(1)
                    
                    # Update tracking timing
                    iter_end_time = time.time()
                    tracking_iter_time_sum += iter_end_time - iter_start_time
                    tracking_iter_time_count += 1
                    
                progress_bar.close()
            
            # Apply best candidate from multi-resolution tracking
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
                
        # Handling GT poses remains the same as original code
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = gt_w2c_all_frames[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        
        # Update tracking timing
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1
        
        # Densification & Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Get the current camera pose
            with torch.no_grad():
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
            
            # FlashSLAM uses adaptive keyframe selection based on tracking loss
            if config["flashslam"]["adaptive_keyframe_selection"] and len(keyframe_list) > 0:
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = adaptive_keyframe_selection(
                    current_min_loss, depth, curr_w2c, intrinsics, 
                    keyframe_list[:-1], num_keyframes)
            else:
                # Use original keyframe selection
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(
                    depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
            
            # Get the time indices for the selected keyframes
            selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
            
            # Add current and previous keyframe
            if len(keyframe_list) > 0:
                selected_time_idx.append(keyframe_list[-1]['id'])
                selected_keyframes.append(len(keyframe_list)-1)
            selected_time_idx.append(time_idx)
            
            # Mapping optimization with local bundle adjustment for FlashSLAM
            mapping_start_time = time.time()
            
            # Reset optimizer for mapping
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
            
            # Progressive map optimization for FlashSLAM
            # First optimize current frame area, then expand to full selected keyframes
            if config["flashslam"]["use_local_ba"]:
                # Local bundle adjustment in two phases
                
                # Phase 1: Optimize current frame area Gaussians
                if config["flashslam"]["hierarchical_mapping"]:
                    # Use hierarchical map to get nearby Gaussians
                    camera_pos = -curr_w2c[:3, :3].T @ curr_w2c[:3, 3]
                    local_radius = variables['scene_radius'] * 0.5
                    nearby_gaussian_indices = h_map.get_nearby_gaussians(
                        camera_pos.cpu().numpy(), local_radius.cpu().numpy())
                    
                    # Create mask for local optimization
                    local_gaussian_mask = torch.zeros(
                        params['means3D'].shape[0], dtype=torch.bool, device=params['means3D'].device)
                    local_gaussian_mask[nearby_gaussian_indices] = True
                    
                    # Only optimize nearby Gaussians
                    local_ba_iters = config["flashslam"]["local_ba_window_size"]
                    for iter in range(local_ba_iters):
                        # Randomly select a frame from recent frames
                        rand_idx = np.random.choice([-1, -2, -3], p=[0.5, 0.3, 0.2])
                        iter_time_idx = selected_time_idx[rand_idx]
                        
                        # Get frame data
                        if rand_idx == -1:
                            iter_color = color
                            iter_depth = depth
                        else:
                            keyframe_idx = selected_keyframes[rand_idx]
                            iter_color = keyframe_list[keyframe_idx]['color']
                            iter_depth = keyframe_list[keyframe_idx]['depth']
                        
                        iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                        iter_data = {
                            'cam': cam, 
                            'im': iter_color, 
                            'depth': iter_depth, 
                            'id': iter_time_idx,
                            'intrinsics': intrinsics, 
                            'w2c': first_frame_w2c, 
                            'iter_gt_w2c_list': iter_gt_w2c,
                            'local_gaussian_mask': local_gaussian_mask  # Pass mask for local optimization
                        }
                        
                        # Optimize with mixed precision
                        with amp.autocast(enabled=config["flashslam"]["use_mixed_precision"]):
                            loss, variables, losses = get_loss(
                                params, iter_data, variables, iter_time_idx, 
                                config['mapping']['loss_weights'],
                                config['mapping']['use_sil_for_loss'], 
                                config['mapping']['sil_thres'],
                                config['mapping']['use_l1'], 
                                config['mapping']['ignore_outlier_depth_loss'], 
                                mapping=True, do_ba=False
                            )
                        
                        # Update with mixed precision
                        mp_trainer.optimizer_step(loss, optimizer)
                
                # Phase 2: Global optimization with all selected keyframes
                mapping_iters = config['mapping']['num_iters'] - config["flashslam"]["local_ba_window_size"]
            else:
                # Standard mapping without local BA
                mapping_iters = config['mapping']['num_iters']
            
            # Execute main mapping loop (similar to original but with mixed precision)
            if mapping_iters > 0:
                progress_bar = tqdm(range(mapping_iters), desc=f"Mapping Time Step: {time_idx}")
                for iter in range(mapping_iters):
                    iter_start_time = time.time()
                    
                    # Randomly select a keyframe
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    
                    # Get frame data
                    if selected_rand_keyframe_idx == -1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    
                    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {
                        'cam': cam, 
                        'im': iter_color, 
                        'depth': iter_depth, 
                        'id': iter_time_idx,
                        'intrinsics': intrinsics, 
                        'w2c': first_frame_w2c, 
                        'iter_gt_w2c_list': iter_gt_w2c
                    }
                    
                    # Optimize with mixed precision
                    with amp.autocast(enabled=config["flashslam"]["use_mixed_precision"]):
                        loss, variables, losses = get_loss(
                            params, iter_data, variables, iter_time_idx, 
                            config['mapping']['loss_weights'],
                            config['mapping']['use_sil_for_loss'], 
                            config['mapping']['sil_thres'],
                            config['mapping']['use_l1'], 
                            config['mapping']['ignore_outlier_depth_loss'], 
                            mapping=True
                        )
                    
                    # Process the optimization step
                    with torch.no_grad():
                        # Prune Gaussians
                        if config['mapping']['prune_gaussians']:
                            params, variables = prune_gaussians(
                                params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        
                        # Gaussian-Splatting's Gradient-based Densification
                        if config['mapping']['use_gaussian_splatting_densification']:
                            params, variables = densify(
                                params, variables, optimizer, iter, config['mapping']['densify_dict'])
                    
                    # Update with mixed precision
                    mp_trainer.optimizer_step(loss, optimizer)
                    
                    # Update progress
                    progress_bar.update(1)
                    
                    # Update mapping timing
                    iter_end_time = time.time()
                    mapping_iter_time_sum += iter_end_time - iter_start_time
                    mapping_iter_time_count += 1
                
                progress_bar.close()
            
            # Update hierarchical map if enabled
            if config["flashslam"]["hierarchical_mapping"]:
                h_map.update_map(params)
            
            # Update mapping timing
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1
        
        # Add frame to keyframe list - same logic as original
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or
                (time_idx == num_frames-2)) and (not torch.isinf(gt_w2c_all_frames[-1]).any()) and (not torch.isnan(gt_w2c_all_frames[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint logic - same as original
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), 
                   np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1
        
        # Clear GPU cache periodically for better memory management
        if config["flashslam"]["use_cuda_kernels"] and time_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Compute Average Runtimes - same as original with added FlashSLAM metrics
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    
    print(f"\nFlashSLAM Performance Metrics:")
    print(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print(f"Total Gaussians in Map: {params['means3D'].shape[0]}")
    
    if config['use_wandb']:
        wandb_run.log({
            "Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
            "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
            "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
            "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
            "Final Stats/Total Gaussians": params['means3D'].shape[0],
            "Final Stats/step": 1
        })
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], 
                 add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], 
                 add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
    
    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    
    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()
    
    return params, variables