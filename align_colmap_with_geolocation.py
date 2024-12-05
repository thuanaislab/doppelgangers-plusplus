import numpy as np
import sqlite3
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from utils.colmap_utils import read_images_binary
import open3d as o3d

def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose) 


def read_colmap_positions(model_path):
    images = read_images_binary(model_path+"/images.bin")
    positions = {}
    for k,v in images.items():
        colmap_extrinsics = np.concatenate((v.qvec, v.tvec))
        extrinsics = pose_from_qwxyz_txyz(colmap_extrinsics)
        positions[k] = extrinsics[:3, 3]
    return positions


def read_geolocations(database_path):
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, prior_tx, prior_ty, prior_tz FROM images WHERE prior_tx IS NOT NULL")
        geolocations = {row[0]: np.array([row[1], row[2], row[3]]) for row in cursor.fetchall()}
    return geolocations

    
def estimate_similarity_ransac(colmap_positions, geolocations):
    common_ids = set(colmap_positions.keys()).intersection(geolocations.keys())
    if not common_ids:
        raise ValueError("No common image IDs between COLMAP positions and geolocations.")

    colmap_points = np.array([colmap_positions[image_id] for image_id in common_ids])
    geo_points = np.array([geolocations[image_id] for image_id in common_ids])

    ransac = RANSACRegressor(min_samples=3, residual_threshold=30.0) # make sure parameters are consistent when evaluating different reconstructions of the same scene 
    ransac.fit(colmap_points, geo_points)

    inliers = ransac.inlier_mask_
    colmap_inliers = colmap_points[inliers]
    geo_inliers = geo_points[inliers]

    colmap_centered = colmap_inliers - np.mean(colmap_inliers, axis=0)
    geo_centered = geo_inliers - np.mean(geo_inliers, axis=0)

    H = colmap_centered.T @ geo_centered
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T


    scale = np.sum(S) / np.sum(colmap_centered**2)
    translation = np.mean(geo_inliers, axis=0) - scale * rotation @ np.mean(colmap_inliers, axis=0)

    # Calculate inlier ratio
    inlier_ratio = np.sum(inliers) / len(common_ids)

    # Calculate inlier error
    transformed_inliers = scale * colmap_inliers @ rotation.T + translation
    inlier_error = np.mean(np.linalg.norm(transformed_inliers - geo_inliers, axis=1))

    return scale, rotation, translation, inlier_ratio, inlier_error, colmap_points, geo_points



def apply_similarity_transform(points, scale, rotation, translation):
    return scale * points @ rotation.T + translation

def save_points_as_ply(points_3d, filename='points.ply'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"File saved as {filename}")


if __name__ == "__main__":
    model_path = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/verify/before_disambiguate"
    database_path = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/verify/working_database.db"

    colmap_positions = read_colmap_positions(model_path)
    geolocations = read_geolocations(database_path)

    scale, rotation, translation, inlier_ratio, inlier_error, colmap_points, geo_points = estimate_similarity_ransac(colmap_positions, geolocations)

    print("Estimated Transformation:")
    print(f"Scale: {scale}")
    print(f"Rotation:\n{rotation}")
    print(f"Translation: {translation}")
    print(f"Inlier Ratio: {inlier_ratio}")
    print(f"Inlier Error: {inlier_error}")

    transformed_points = apply_similarity_transform(colmap_points, scale, rotation, translation)
    save_points_as_ply(transformed_points, filename='transformed_points.ply')
    save_points_as_ply(geo_points, filename='geolocations.ply')

