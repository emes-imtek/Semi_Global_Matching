import cv2
import numpy as np
import yaml
from PIL import Image
import os
import random
import tifffile
from tqdm import tqdm
import time
import json
import logging

# Constants
INVALID_DEPTH = -1  # Define invalid depth value
GT_DEPTH_INVALID = 65504  # Blender/EXR "no depth" sentinel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_images(left_path, right_path):
    """
    Loads images, applies RGBA to mask for transparency, and converts to BGR.
    """
    logger.info(f"Loading images: {left_path}, {right_path}")
    left_img_pil = Image.open(left_path).convert("RGBA")
    right_img_pil = Image.open(right_path).convert("RGBA")
    left_mask = np.array(left_img_pil)[:, :, 3] > 0
    right_mask = np.array(right_img_pil)[:, :, 3] > 0
    left_img = cv2.cvtColor(np.array(left_img_pil)[:, :, :3], cv2.COLOR_RGB2BGR)
    right_img = cv2.cvtColor(np.array(right_img_pil)[:, :, :3], cv2.COLOR_RGB2BGR)
    return left_img, right_img, left_mask, right_mask

def load_calibration(calib_path):
    """
    Loads calibration matrices from a YAML file.
    """
    logger.info(f"Loading calibration from {calib_path}")
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    K1 = fs.getNode('M1').mat()
    K2 = fs.getNode('M2').mat()
    T = fs.getNode('T').mat()
    T = T * 0.001  # Convert baseline to meters if needed

    # Transpose T if necessary (make sure it's a 3x1 column vector)
    if T.shape == (1, 3):
        T = T.T  # Transpose

    fs.release()
    logger.info("Calibration matrices loaded successfully.")
    return K1, K2, T

def compute_disparity(left_img, right_img):
    """
    Computes the disparity map between the stereo images.
    """
    logger.info("Computing disparity map...")
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=320,  # adjust to your scene if needed
        blockSize=7,
        P1=8*3*7**2,
        P2=32*3*7**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    logger.info("Disparity map computed successfully.")
    return disparity

def compute_depth_manual(disparity, focal_length_px, baseline_mm):
    """
    Computes depth map from the disparity map using Z = f * B / d.
    Returns depth in mm (to match Blender GT).
    """
    logger.info("Computing depth map from disparity (manual formula)...")
    with np.errstate(divide='ignore'):
        depth = (focal_length_px * baseline_mm) / disparity
        depth[(disparity <= 0) | np.isnan(depth) | np.isinf(depth)] = INVALID_DEPTH
    logger.info("Depth map computed successfully (manual).")
    return depth

def save_depth(depth, valid_mask, output_path_png, output_path_tiff):
    """
    Saves depth map to both PNG (visualized) and TIFF (raw) formats.
    """
    logger.info(f"Saving depth map to {output_path_tiff} and {output_path_png}...")
    depth[valid_mask == 0] = INVALID_DEPTH
    tifffile.imwrite(output_path_tiff, depth.astype(np.float32))

    # Visualization for PNG (optional)
    valid = (depth != INVALID_DEPTH)
    if np.any(valid):
        depth_vis = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis[~valid] = 0
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_color[~valid] = 0
        cv2.imwrite(output_path_png, depth_color)
    logger.info("Depth map saved successfully.")

def save_point_cloud(disparity, left_img, valid_mask, focal_length_px, baseline_mm, ply_path):
    """
    Saves the 3D point cloud to a PLY file using the manual depth calculation.
    """
    logger.info(f"Saving point cloud to {ply_path}...")
    h, w = disparity.shape
    Q = np.float32([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, focal_length_px],
        [0, 0, 1/baseline_mm, 0]
    ])
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    mask = (disparity > 0) & (valid_mask > 0) & np.all(np.isfinite(points_3D), axis=2)
    out_points = points_3D[mask]
    out_colors = colors[mask]
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(out_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(out_points, out_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    logger.info("Point cloud saved successfully.")

def find_image_pairs(root):
    """
    Finds corresponding left and right images in the directory structure.
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        lefts = [f for f in filenames if f.endswith("_L.png")]
        for lf in lefts:
            base = lf[:-6]
            rf = base + "_R.png"
            if rf in filenames:
                pairs.append((os.path.join(dirpath, lf), os.path.join(dirpath, rf)))
    return pairs

def load_ground_truth_depth(npz_path):
    """
    Loads the ground truth depth map from a .npz file.
    Assumes the depth map is stored with key 'depth_map', 'depth', or 'arr_0'.
    """
    try:
        data = np.load(npz_path)
        for key in ['depth_map', 'depth', 'arr_0']:
            if key in data:
                return data[key]
        raise KeyError(f"No depth map key found in {npz_path}. Available keys: {list(data.keys())}")
    except Exception as e:
        logger.warning(f"Failed to load ground truth from {npz_path}: {e}")
        return None

def compute_error_metrics(est_depth, gt_depth, valid_mask):
    """
    Computes MAE and RMSE between estimated and ground truth depth maps.
    Only evaluates at valid pixels.
    Masks out invalid GT pixels (typically 65504 for Blender/EXR/float16 output).
    """
    valid = (valid_mask > 0) & np.isfinite(gt_depth) & (gt_depth != INVALID_DEPTH) & (est_depth != INVALID_DEPTH) & (gt_depth < GT_DEPTH_INVALID)
    if not np.any(valid):
        return None, None, 0
    diff = est_depth[valid] - gt_depth[valid]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse, np.count_nonzero(valid)

def print_unit_check(gt_depth, est_depth):
    """
    Print statistics for quick unit check.
    """
    gt_valid = gt_depth[(gt_depth > 0) & np.isfinite(gt_depth) & (gt_depth < GT_DEPTH_INVALID)]
    est_valid = est_depth[(est_depth > 0) & np.isfinite(est_depth)]
    logger.info(f"GT depth min/max: {gt_valid.min() if gt_valid.size > 0 else 'n/a'} / {gt_valid.max() if gt_valid.size > 0 else 'n/a'}")
    logger.info(f"Est depth min/max: {est_valid.min() if est_valid.size > 0 else 'n/a'} / {est_valid.max() if est_valid.size > 0 else 'n/a'}")
    logger.info(f"Sample GT: {gt_valid[:5]}")
    logger.info(f"Sample Est: {est_valid[:5]}")

def main():
    """
    Main function to load images, compute disparity, and save results.
    """
    input_root = '/media/nvidiapc/Data/Datasets/Blender_Pruefmuster_Andreas/Ground_Truth_Small'
    output_root = '/media/nvidiapc/Data/Datasets/Blender_Pruefmuster_Andreas/SGM_Estimation'
    calib_path = '/media/nvidiapc/Data/Datasets/Blender_Pruefmuster_Andreas/Ground_Truth/endoscope_calibration_Blender_para.yaml'
    
    K1, K2, T = load_calibration(calib_path)
    focal_length_px = K1[0, 0]
    baseline_mm = abs(T[0, 0])  # Make sure baseline is positive

    pairs = find_image_pairs(input_root)
    sampled_indices = set(random.sample(range(len(pairs)), min(5, len(pairs))))

    fps_summary = {}
    total_frames = 0
    total_time = 0.0

    eval_results = {}

    for i, (left_path, right_path) in enumerate(tqdm(pairs, desc="Processing pairs")):
        rel_dir = os.path.relpath(os.path.dirname(left_path), input_root)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        left_img, right_img, left_mask, right_mask = load_images(left_path, right_path)
        # No rectification for Blender data
        left_rect = left_img
        right_rect = right_img
        left_mask_rect = left_mask.astype(np.uint8) * 255
        right_mask_rect = right_mask.astype(np.uint8) * 255

        start_time = time.time()
        disparity = compute_disparity(left_rect, right_rect)
        elapsed_time = time.time() - start_time

        valid_mask = cv2.bitwise_and(left_mask_rect, right_mask_rect)

        basename = os.path.basename(left_path).replace(".png", "")
        out_disp_path_png = os.path.join(out_dir, f"{basename}_disp.png")
        out_disp_path_tiff = os.path.join(out_dir, f"{basename}.tiff")

        # Compute the depth map from disparity, in mm to match Blender GT
        depth = compute_depth_manual(disparity, focal_length_px, baseline_mm)  # returns mm

        logger.info(f"Depth range: min={np.min(depth)}, max={np.max(depth)}")

        # Save the depth map (instead of disparity)
        save_depth(depth, valid_mask, out_disp_path_png, out_disp_path_tiff)

        # --- Evaluation against GT depth map (.npz) ---
        gt_path = left_path.replace(".png", ".npz")
        mae, rmse, n_valid = None, None, 0
        if os.path.exists(gt_path):
            gt_depth = load_ground_truth_depth(gt_path)
            if gt_depth is not None:
                # Print unit check for the first image
                if i == 0:
                    print_unit_check(gt_depth, depth)
                mae, rmse, n_valid = compute_error_metrics(depth, gt_depth, valid_mask)
                logger.info(f"[EVAL] {basename}: MAE={mae:.6f} mm, RMSE={rmse:.6f} mm ({n_valid} valid px)")
                eval_results.setdefault(rel_dir, []).append({
                    "file_name": os.path.basename(left_path).replace(".png", ".tiff"),
                    "mae": float(mae) if mae is not None else None,
                    "rmse": float(rmse) if rmse is not None else None,
                    "n_valid_px": int(n_valid)
                })
            else:
                logger.warning(f"No ground truth depth loaded for {basename}")
        else:
            logger.warning(f"No ground truth file: {gt_path}")

        subfolder = rel_dir
        fps_summary.setdefault(subfolder, {"total_time": 0.0, "frames": 0})
        fps_summary[subfolder]["total_time"] += elapsed_time
        fps_summary[subfolder]["frames"] += 1
        total_time += elapsed_time
        total_frames += 1

        if i in sampled_indices:
            ply_dir = os.path.join(output_root, "sample_plys")
            os.makedirs(ply_dir, exist_ok=True)
            ply_name = os.path.basename(left_path).replace("_L.png", ".ply")
            ply_path = os.path.join(ply_dir, ply_name)
            save_point_cloud(disparity, left_rect, valid_mask, focal_length_px, baseline_mm, ply_path)

    # --- FPS summary ---
    fps_results = {"subfolders": [], "overall_average_fps": 0, "total_frames_processed": total_frames}
    for subfolder, stats in fps_summary.items():
        frames = stats["frames"]
        time_sec = stats["total_time"]
        fps = frames / time_sec if time_sec > 0 else 0
        fps_results["subfolders"].append({
            "subfolder": subfolder,
            "average_fps": fps,
            "frames": frames
        })
    if total_time > 0:
        fps_results["overall_average_fps"] = total_frames / total_time

    with open(os.path.join(output_root, "fps_summary.json"), "w") as f:
        json.dump(fps_results, f, indent=4)

    # --- Save evaluation summary ---
    summary = {"folders": [], "total_average_mae": 0, "total_average_rmse": 0, "n_files": 0}
    total_mae, total_rmse, total_count = 0, 0, 0
    for folder, files in eval_results.items():
        folder_mae = np.mean([f["mae"] for f in files if f["mae"] is not None])
        folder_rmse = np.mean([f["rmse"] for f in files if f["rmse"] is not None])
        summary["folders"].append({
            "folder": folder,
            "file_pairs": len(files),
            "folder_average_mae": folder_mae,
            "folder_average_rmse": folder_rmse,
            "files": files
        })
        total_mae += folder_mae * len(files)
        total_rmse += folder_rmse * len(files)
        total_count += len(files)
    if total_count > 0:
        summary["total_average_mae"] = total_mae / total_count
        summary["total_average_rmse"] = total_rmse / total_count
        summary["n_files"] = total_count
    with open(os.path.join(output_root, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    main()