import os
import json
import sqlite3
import math
import subprocess
import shutil

from utils.database import pair_id_to_image_ids
from utils.colmap_utils import read_images_binary

def lla_to_ecef(lat, lon, alt):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f ** 2  # Square of eccentricity

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Compute the prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)

    # Compute ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)

    return x, y, z


def extract_image_id(filename):
    return filename.split('_')[0].split('.')[0]


def insert_ecef_data(database_path, image_dir, metadata_dir):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, name FROM images")
    image_rows = cursor.fetchall()
    image_map = {row[1]: row[0] for row in image_rows}  

    for image_filename in os.listdir(image_dir):
        if not image_filename.endswith(('.jpg', '.png')):
            continue

        image_id = extract_image_id(image_filename)
        metadata_file = os.path.join(metadata_dir, f"{image_id}.json")

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract GPS data
            coordinates = metadata.get("geometry", {}).get("coordinates")
            altitude = 0 #ignore altitude, or metadata.get("altitude")

            if coordinates and altitude is not None:
                lat, lon = coordinates
                alt = altitude
                x, y, z = lla_to_ecef(lat, lon, alt)
                db_image_id = image_map.get(image_filename)
                if db_image_id:
                    cursor.execute("""
                        UPDATE images
                        SET prior_tx = ?, prior_ty = ?, prior_tz = ?
                        WHERE image_id = ?
                    """, (x, y, z, db_image_id))
                    print(f"Updated ECEF priors for {image_filename}: x={x}, y={y}, z={z}")

    conn.commit()
    conn.close()


def feature_extraction_and_matching(
    database_path, image_dir, metadata_dir, colmap_bin_path="colmap"
):
    print("Running feature extraction for new images...")
    feature_extractor_cmd = [
        colmap_bin_path, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
    ]
    subprocess.run(feature_extractor_cmd, check=True)

    print("Inserting ECEF priors into the database...")
    insert_ecef_data(database_path, image_dir, metadata_dir)

    print("Running image matching...")
    os.makedirs('weights', exist_ok=True)
    vocab_tree_path = 'weights/vocab_tree_flickr100K_words1M.bin'
    if not os.path.exists(vocab_tree_path):
        subprocess.run(
            ["wget", "https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin", "-P", "weights/"],
            check=True
        )
    matcher_cmd = [
        colmap_bin_path, "vocab_tree_matcher",
        "--database_path", database_path,
        "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
    ]
    subprocess.run(matcher_cmd, check=True)
    
    
def register_geo_images(
    model_path, database_path, output_path, colmap_bin_path="colmap"
):
    os.makedirs(output_path, exist_ok=True)
    print("Registering geo-tagged images using image_registrator...")
    image_registrator_cmd = [
        colmap_bin_path, "image_registrator",
        "--database_path", database_path,
        "--input_path", model_path,
        "--output_path", output_path,
    ]
    subprocess.run(image_registrator_cmd, check=True)

    print(f"Registration complete. Model saved at: {output_path}")


def get_valid_image_ids(new_reconstruction_path):
    images_bin_path = os.path.join(new_reconstruction_path, "images.bin")
    images = read_images_binary(images_bin_path) 
    return set(images.keys())


def remove_pruned_matches(database_path, valid_image_ids, start_idx):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT pair_id FROM matches")
    all_pairs = cursor.fetchall()
    invalid_pair_ids = []
    for (pair_id,) in all_pairs:
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        if (image_id1 not in valid_image_ids and image_id1 < start_idx) or (image_id2 not in valid_image_ids and image_id2 < start_idx):
            invalid_pair_ids.append(pair_id)

    if invalid_pair_ids:
        cursor.executemany(
            "DELETE FROM matches WHERE pair_id = ?",
            [(pair_id,) for pair_id in invalid_pair_ids]
        )
        conn.commit()
        print(f"Removed {len(invalid_pair_ids)} invalid matches from the database.")
    else:
        print("No invalid matches to remove.")

    conn.close()


def find_largest_image_id(database_path):
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Query to find the largest image_id
        cursor.execute("SELECT MAX(image_id) FROM images")
        result = cursor.fetchone()
        
        if result and result[0] is not None:
            largest_image_id = result[0]
            return largest_image_id
        else:
            print("No images found in the database.")
            return 0
    except sqlite3.Error as e:
        print(f"Error accessing database: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    old_model_path = "/share/phoenix/nfs04/S7/rc844/SFM3D/reconstruct/disambiguity_dataset/images/alexander_nevsky_cathedral/sparse_color/0"  # Path to existing model
    database_path = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/dbs/database.db"  # Path to COLMAP database
    image_dir = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/images_undistort"
    metadata_dir = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/image_infos"
    output_path = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/verify"  # Output directory for updated model
    new_model_path = "/share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/mast3r+dopp/sparse_doppelgangers_0.800/0"  # Path to new model

    os.makedirs(output_path, exist_ok=True)
    
    # geo verify original model    
    working_db_path = os.path.join(output_path, "working_database.db")
    if not os.path.exists(working_db_path):
        shutil.copy(database_path, working_db_path)
    feature_extraction_and_matching(working_db_path, image_dir, metadata_dir)
    register_geo_images(old_model_path, working_db_path, output_path+'/before_disambiguate')

    # geo verify disambiguated model
    start_idx = find_largest_image_id(working_db_path) + 1
    new_working_db_path = os.path.join(output_path, "working_database_disambiguate.db")
    if not os.path.exists(new_working_db_path):
        shutil.copy(working_db_path, new_working_db_path)
    valid_image_ids = get_valid_image_ids(new_model_path)
    remove_pruned_matches(new_working_db_path, valid_image_ids, start_idx)
    register_geo_images(new_model_path, new_working_db_path, output_path+'/after_disambiguate')

    