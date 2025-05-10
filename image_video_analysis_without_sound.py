import cv2
import json
import numpy as np
import pandas as pd
from deepface import DeepFace
from skimage.transform import resize
from skimage.feature import hog
from ultralytics import YOLO
from textblob import TextBlob
import lzma  # Import LZMA module for .xz files
import random
import ffmpeg
from tqdm import tqdm

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load YOLO model
yolo_model = YOLO("yolov8l.pt")


# Define metadata fields
METADATA_FIELDS = [
    "filename", "__typename", "comments_disabled", "height", "width", "display_url",
    "likes", "caption", "comments", "id", "shortcode",
    "timestamp", "thumbnail_src", "paid_partnership", "coauthor_accounts"
]


# Define image/video analysis columns
FEATURE_COLUMNS = [
    "filename", "person_count", "object_count", "detected_objects", "age", "gender",
    "facial_expression", "emotions", "smiling (%)", "magnitude", "theta",
    "brightness", "contrast", "saturation", "RGB", "no of lines",
    "no of parallel lines", "no of corners", "Histogram of Oriented Gradients (HOG)",
    "dominant_color", "avg_object_size_ratio (%)", "hair_color", "baldness",
    "foreground_motion", "warm_hue_propagation",
    "video length"
]

# Final expected columns
COLUMNS = FEATURE_COLUMNS + METADATA_FIELDS[1:]

# Function to extract JSON metadata, handling .json.xz compressed files
def extract_json_metadata(path, filename):
    json_path = os.path.join(path, filename)
    xz_path = json_path + ".xz"  # Try checking for .json.xz version

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            myjson = json.load(f)
    elif os.path.exists(xz_path):  # Check if .json.xz exists
        with lzma.open(xz_path, "rt") as f:  # Open compressed .xz file
            myjson = json.load(f)
    else:
        print(f"Warning: Metadata file not found: {json_path} or {xz_path}")
        return {key: np.nan for key in METADATA_FIELDS}  # Return NaN values if no metadata is found

    node = myjson.get("node", {})

    iphone_struct = node.get("iphone_struct", {})

    return {
        "filename": filename,
        "__typename": node.get("__typename", np.nan),
        "comments_disabled": node.get("comments_disabled", np.nan),
        "height": node.get("dimensions", {}).get("height", np.nan),
        "width": node.get("dimensions", {}).get("width", np.nan),
        "display_url": node.get("display_url", np.nan),
        "likes": node.get("edge_media_preview_like", {}).get("count", np.nan),
        "caption": node.get("edge_media_to_caption", {}).get("edges", np.nan),
        "comments": node.get("edge_media_to_comment", {}).get("count", np.nan),
        "id": node.get("id", np.nan),
        "shortcode": node.get("shortcode", np.nan),
        "timestamp": node.get("taken_at_timestamp", np.nan),
        "thumbnail_src": node.get("thumbnail_src", np.nan),
        "paid_partnership": iphone_struct.get("is_paid_partnership", False),
        "coauthor_accounts": iphone_struct.get("coauthor_producers", np.nan)
    }



# Function to detect objects in an image
from collections import Counter

def detect_objects(results):
    detected_objects = []
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls < len(yolo_model.names):
            detected_objects.append(yolo_model.names[cls])

    # Count occurrences of each object type
    object_counts = dict(Counter(detected_objects))

    person_count = object_counts.get("person", 0)

    # Convert to JSON string
    detected_objects_json = json.dumps(object_counts)

    return person_count, detected_objects_json


from keras import backend as K
# Detect emotions & facial expressions
def analyze_facial_features(image):
    try:
        results = DeepFace.analyze(
            img_path=image,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            detector_backend='opencv'  # or 'skip' if you're handling face detection yourself
        )

        K.clear_session()

        if not isinstance(results, list):
            results = [results]  # Ensure it's a list

        ages = []
        gender_counts = {"man": 0, "woman": 0}
        emotion_counts = {}
        smile_scores = []

        for result in results:
            # Collect age
            age = result.get("age", None)
            if age is not None:
                ages.append(age)

            # Use dominant gender directly
            gender = result.get("dominant_gender", None)
            if gender:
                if gender.lower() == "man":
                    gender_counts["man"] += 1
                elif gender.lower() == "woman":
                    gender_counts["woman"] += 1

            # Collect emotion and count occurrences
            emotion = result.get("dominant_emotion", None)
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            emotions = result.get("emotion", {})
            happy_score = emotions.get("happy", None)
            if happy_score is not None:
                smile_scores.append(happy_score)


        # Average age calculation
        avg_age = np.mean(ages) if ages else np.nan

        # Normalize emotions to percentages
        if emotion_counts:
            total_count = sum(emotion_counts.values())
            emotion_percentages = {k: round((v / total_count) * 100, 1) for k, v in emotion_counts.items()}
            emotions_summary = json.dumps(emotion_percentages)  # Save as JSON string
        else:
            emotions_summary = None

        # Format gender counts as a single string
        gender_count_str = f"man: {gender_counts['man']}, woman: {gender_counts['woman']}"

        avg_smile_intensity = np.mean(smile_scores) if smile_scores else np.nan

        return (
            avg_age,
            gender_count_str,
            max(emotion_counts, key=emotion_counts.get) if emotion_counts else None,
            emotions_summary,  # Store as JSON string
            avg_smile_intensity
        )

    except Exception as e:
        print(f"Error analyzing facial features: {e}")
        return np.nan, "man: 0, woman: 0", None, None, np.nan

def detect_smile_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    smile_ratios = []

    for (x, y, w, h) in faces:
        face_area = w * h
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
        )

        if len(smiles) > 0:
            # 가장 큰 입 기준
            largest_smile = max(smiles, key=lambda s: s[2] * s[3])
            sx, sy, sw, sh = largest_smile
            smile_area = sw * sh
            smile_ratio = (smile_area / face_area) * 100
            smile_ratios.append(smile_ratio)

        print('smile: ', smile_ratios)

    return np.mean(smile_ratios) if smile_ratios else np.nan


# Compute optical flow (motion estimation)
def compute_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude.mean(), theta.mean()

# Count corners in an image
def count_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    return np.count_nonzero(dst > 0.01 * dst.max())

def calculate_object_size(results, image):
    total_object_size_ratio = 0
    object_count = 0

    # Get total image size (width * height)
    image_height, image_width = image.shape[:2]
    total_image_size = image_width * image_height

    if total_image_size == 0:
        return 0, 0

    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]  # Bounding box coordinates

        # Convert tensors to float values
        w = float(x2 - x1)
        h = float(y2 - y1)
        object_size = w * h

        # Convert object size to percentage of total image size
        object_size_ratio = (object_size / total_image_size) * 100

        total_object_size_ratio += object_size_ratio
        object_count += 1

    return total_object_size_ratio, object_count

# Detect hair color and baldness using yolo
def detect_hair_color_with_deepface(image):
    try:
        # Use DeepFace to detect face and get bounding box
        results = DeepFace.analyze(image, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        if not isinstance(results, list):
            results = [results]  # Ensure it's a list

        avg_hue_values = []
        baldness_ratios = []
        detected_faces = 0

        for result in results:
            region = result.get("region", {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w == 0 or h == 0:
                continue

            # Estimate hair region above the face
            hair_x1 = max(x - int(w * 0.1), 0)
            hair_x2 = min(x + w + int(w * 0.1), image.shape[1])
            hair_y1 = max(y - int(h * 0.5), 0)  # Top of the face
            hair_y2 = y

            hair_region = image[hair_y1:hair_y2, hair_x1:hair_x2]
            if hair_region.size == 0:
                continue

            hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)

            # Relax thresholds for dark hair
            hair_mask = (hsv[:, :, 1] > 10) & (hsv[:, :, 2] > 30)

            # Backup for black hair detection
            if np.sum(hair_mask) == 0:
                black_hair = hsv[:, :, 2] < 30
                if np.sum(black_hair) > 0:
                    avg_hue = 0  # Black hair hue is near 0
                    hair_size = np.sum(black_hair)
                else:
                    hair_size = 0
            else:
                avg_hue = np.mean(hsv[:, :, 0][hair_mask])
                hair_size = np.sum(hair_mask)

            # If hair is detected, calculate baldness
            face_size = w * h
            baldness_ratio = np.nan if hair_size == 0 else 1 - (hair_size / face_size)

            if not np.isnan(avg_hue):
                avg_hue_values.append(avg_hue)

            baldness_ratios.append(baldness_ratio)
            detected_faces += 1

        # Compute the average across all detected faces
        avg_hue = np.mean(avg_hue_values) if avg_hue_values else np.nan
        avg_baldness = np.mean(baldness_ratios) if baldness_ratios else np.nan

        return avg_hue, avg_baldness

    except Exception as e:
        print(f"Failed to analyze hair with DeepFace: {e}")
        return np.nan, np.nan

# Detect foreground motion using frame differencing
def detect_foreground_motion(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.sum(diff)

# Extract warm hue propagation
def warm_hue_propagation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.sum(hsv[:, :, 0] > 20)

# Function to compute HOG features
def compute_hog(image):
    resized_img = resize(image, (128, 64))
    fd, _ = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    return len(fd)

# Function to detect lines and parallel lines
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=50, maxLineGap=20)

    if lines is not None:
        num_lines = len(lines)
        num_parallel_lines = sum(
            1 for line in lines
            if abs(line[0][0] - line[0][2]) < 5  # Small threshold for parallelism
        )
    else:
        num_lines = 0
        num_parallel_lines = 0

    return num_lines, num_parallel_lines

from sklearn.cluster import KMeans
# Function to detect dominant color
def dominant_color(image):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

# Function to extract brightness, contrast, and saturation
def extract_brightness_contrast_saturation(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(yuv)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return y.mean(), y.std(), hsv[:, :, 1].mean()

def calculate_average_rgb_histogram(frames):
    red_vals = []
    green_vals = []
    blue_vals = []

    for frame in frames:
        r, g, b = cv2.split(frame)
        red_vals.append(r.mean())
        green_vals.append(g.mean())
        blue_vals.append(b.mean())

    avg_r = np.mean(red_vals) if red_vals else np.nan
    avg_g = np.mean(green_vals) if green_vals else np.nan
    avg_b = np.mean(blue_vals) if blue_vals else np.nan

    return avg_r, avg_g, avg_b

def compute_line_orientation_histogram(lines):
    if lines is not None:
        orientations = []
        for line in lines:
            if isinstance(line, (list, np.ndarray)) and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]
                orientation = np.arctan2(y2 - y1, x2 - x1)
                orientations.append(orientation)

        if orientations:
            orientation_histogram, _ = np.histogram(orientations, bins=36, range=(-np.pi, np.pi))
            return orientation_histogram
    return np.zeros(36)


def compute_line_distance_histogram(lines):
    if lines is not None:
        distances = []
        for line in lines:
            if isinstance(line, (list, np.ndarray)) and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(distance)

        if distances:
            distance_histogram, _ = np.histogram(distances, bins=20, range=(0, max(distances) if distances else 1))
            return distance_histogram
    return np.zeros(20)

# Function to process an image
def process_image(path, filename):
    full_path = os.path.join(path, filename)
    image = cv2.imread(full_path)
    if image is None:
        return [filename] + [np.nan] * (len(FEATURE_COLUMNS) - 1)

    # Single YOLO call
    results = yolo_model(image)

    # Use the same results for all functions
    person_count, detected_objects = detect_objects(results)
    total_object_size_ratio, object_count = calculate_object_size(results, image)
    avg_object_size_ratio = (total_object_size_ratio / object_count) if object_count > 0 else np.nan
    avg_age, gender_count_str, avg_emotion, emotions, _ = analyze_facial_features(image)
    smile_intensity = detect_smile_ratio(image)
    magnitude, theta = np.nan, np.nan
    # smile_ratio = detect_smile(image)

    avg_r, avg_g, avg_b = calculate_average_rgb_histogram([image])

    brightness, contrast, saturation = extract_brightness_contrast_saturation(image)
    num_lines, num_parallel_lines = detect_lines(image)
    no_of_corners = count_corners(image)
    hog_features = compute_hog(image)  # Store length only
    dom_color = dominant_color(image)
    hair_color, baldness = detect_hair_color_with_deepface(image)
    line_orientation_histogram = compute_line_orientation_histogram(detect_lines(image))
    line_distance_histogram = compute_line_distance_histogram(detect_lines(image))
    warm_hue = warm_hue_propagation(image)

    return [
        filename, person_count, object_count, detected_objects, avg_age, gender_count_str, avg_emotion, emotions,
        smile_intensity, magnitude, theta, brightness, contrast, saturation,
        [avg_r, avg_g, avg_b],
        num_lines, num_parallel_lines, no_of_corners,
        hog_features,
        dom_color, avg_object_size_ratio,
        hair_color, baldness, np.nan, warm_hue,
        np.nan, np.nan, np.nan
    ]


#     FEATURE_COLUMNS = [
#     "filename", "person_count", "object_count", "detected_objects", "age", "gender",
#     "facial_expression", "emotions", "smiling (%)", "magnitude", "theta",
#     "brightness", "contrast", "saturation", "RGB", "no of lines",
#     "no of parallel lines", "no of corners", "Histogram of Oriented Gradients (HOG)",
#     "dominant_color", "avg_object_size_ratio (%)", "hair_color", "baldness",
#     "foreground_motion", "warm_hue_propagation", "HSV Saturation",
#     "video length", "Speech Sentiment"
# ]

def process_video(path, filename, mode='sampled'):
    full_path = os.path.join(path, filename)
    try:
        probe = ffmpeg.probe(full_path)
        video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        duration = float(video_stream['duration'])
    except Exception as e:
        print(f"❌ Failed to get video duration from {filename}: {e}")
        duration = np.nan

    # 1. 전체 프레임 추출
    frames = []
    cap = cv2.VideoCapture(full_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)

    # 2. 샘플링 모드 or 전체 프레임 모드 분기
    if mode == 'sampled':
        num_samples = int(total_frames * 0.1)
        valid_start_range = total_frames - 10
        start_indices = random.sample(range(valid_start_range), num_samples)
        sampled_frame_groups = [frames[i:i+2] for i in start_indices if len(frames[i:i+2]) == 2]

    elif mode == 'sequential':
        sampled_frame_groups = [frames[i:i+2] for i in range(total_frames - 1) if len(frames[i:i+2]) == 2]

    else:
        print(f"❌ Unknown mode: {mode}")
        return [filename] + [np.nan] * (len(FEATURE_COLUMNS) - 1)

    # 3. 초기화
    person_counts = []
    detected_objects = []
    object_counts = []
    brightness_vals = []
    contrast_vals = []
    saturation_vals = []
    ages = []
    gender_counts = {"man": 0, "woman": 0}
    emotion_counts = {}
    smile_intensities = []
    dominant_colors = []
    object_size_ratios = []
    hair_colors = []
    baldness_ratios = []
    r_vals, g_vals, b_vals = [], [], []

    # 4. 프레임 분석
    for group in tqdm(sampled_frame_groups, desc="Analyzing frame groups"):
        for frame in group:
            results = yolo_model(frame)
            person_count, detected_objects_per_frame = detect_objects(results)
            person_counts.append(person_count)

            # Convert JSON string to dict
            if detected_objects_per_frame:
                try:
                    frame_objects = json.loads(detected_objects_per_frame)
                    detected_objects.append(list(frame_objects.keys()))
                except:
                    pass

            object_counts.append(len(detected_objects_per_frame))

            try:
                hair_color, baldness = detect_hair_color_with_deepface(frame)
                if not np.isnan(hair_color): hair_colors.append(hair_color)
                if not np.isnan(baldness): baldness_ratios.append(baldness)
            except: pass

            try:
                dom_color = dominant_color(frame)
                dominant_colors.append(dom_color)
            except: pass

            total_object_size_ratio, object_count = calculate_object_size(results, frame)
            if object_count > 0:
                object_size_ratios.append(total_object_size_ratio / object_count)

            avg_age, gender_count_str, avg_emotion, frame_emotions, _ = analyze_facial_features(frame)
            smile_ratio = detect_smile_ratio(frame)
            if not np.isnan(smile_ratio):
                smile_intensities.append(smile_ratio)

            if "man" in gender_count_str:
                gender_counts["man"] += int(gender_count_str.split(",")[0].split(": ")[1])
            if "woman" in gender_count_str:
                gender_counts["woman"] += int(gender_count_str.split(",")[1].split(": ")[1])

            if frame_emotions:
                frame_emotions = json.loads(frame_emotions)
                for emotion, count in frame_emotions.items():
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + count

            if not np.isnan(avg_age): ages.append(avg_age)

            brightness, contrast, saturation = extract_brightness_contrast_saturation(frame)
            brightness_vals.append(brightness)
            contrast_vals.append(contrast)
            saturation_vals.append(saturation)

            r, g, b = cv2.split(frame)
            r_vals.append(r.mean())
            g_vals.append(g.mean())
            b_vals.append(b.mean())

    # 5. Optical Flow 계산
    magnitudes = []
    thetas = []
    motions = []

    for group in sampled_frame_groups:
        for i in range(len(group) - 1):
            frame1, frame2 = group[i], group[i + 1]
            if frame1.shape != frame2.shape:
                continue
            try:
                magnitude, theta = compute_optical_flow(frame1, frame2)
                motion = detect_foreground_motion(frame1, frame2)
                magnitudes.append(magnitude)
                thetas.append(theta)
                motions.append(motion)
            except:
                continue

    # 6. 집계
    avg_magnitude = np.mean(magnitudes) if magnitudes else np.nan
    avg_theta = np.mean(thetas) if thetas else np.nan
    avg_motion = np.mean(motions) if motions else np.nan

    all_detected_objects = [obj for frame_objects in detected_objects for obj in frame_objects]
    detected_objects_json = json.dumps(dict(Counter(all_detected_objects))) if all_detected_objects else None
    most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    emotions_summary = (
        json.dumps({k: round((v / sum(emotion_counts.values())) * 100, 1) for k, v in emotion_counts.items()})
        if emotion_counts else None
    )
    gender_summary = f"man: {gender_counts['man']}, woman: {gender_counts['woman']}"

    avg_person_count = np.mean(person_counts) if person_counts else np.nan
    avg_object_count = np.mean(object_counts) if object_counts else np.nan
    avg_age = np.mean(ages) if ages else np.nan
    avg_smile_intensity = np.mean(smile_intensities) if smile_intensities else np.nan
    avg_brightness = np.mean(brightness_vals) if brightness_vals else np.nan
    avg_contrast = np.mean(contrast_vals) if contrast_vals else np.nan
    avg_saturation = np.mean(saturation_vals) if saturation_vals else np.nan
    avg_r = np.mean(r_vals) if r_vals else np.nan
    avg_g = np.mean(g_vals) if g_vals else np.nan
    avg_b = np.mean(b_vals) if b_vals else np.nan
    avg_hair_color = np.mean(hair_colors) if hair_colors else np.nan
    avg_baldness_ratio = np.mean(baldness_ratios) if baldness_ratios else np.nan
    avg_dom_color = np.mean(dominant_colors, axis=0).tolist() if dominant_colors else [np.nan, np.nan, np.nan]
    avg_object_size_ratio = np.mean(object_size_ratios) if object_size_ratios else np.nan

    return [
        filename, avg_person_count, avg_object_count, detected_objects_json, avg_age, gender_summary,
        most_common_emotion, emotions_summary, avg_smile_intensity, avg_magnitude, avg_theta,
        avg_brightness, avg_contrast, avg_saturation, [avg_r, avg_g, avg_b],
        np.nan, np.nan, np.nan, np.nan, avg_dom_color, avg_object_size_ratio,
        avg_hair_color, avg_baldness_ratio, avg_motion, np.nan,
        duration
    ]


def process_directory(path, model_dir, mode='sampled', save_csv=True):
    data = []

    columns = FEATURE_COLUMNS + METADATA_FIELDS[1:]

    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)

        if ext.lower() in [".jpg", ".jpeg"]:
            row = process_image(path, filename)
        elif ext.lower() == ".mp4":
            row = process_video(path, filename, mode=mode)
        else:
            continue

        import re
        base_name = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC)", filename)
        json_filename = base_name.group(1) + ".json"
        metadata = extract_json_metadata(path, json_filename)

        # Ensure metadata aligns with expected columns
        metadata_values = [metadata.get(col, np.nan) for col in METADATA_FIELDS[1:]]

        row.extend(metadata_values)
        if len(row) != len(columns):
            print(f"❌ Error: Column count mismatch for {filename}")
            print(f"Expected columns: {len(columns)}, Got: {len(row)}")
            print(f"Row Data: {row}")

        # Create single-row DataFrame
        df_row = pd.DataFrame([row], columns=columns)

        # Save incrementally
        csv_name = os.path.join("results", f"{os.path.basename(path)}_analysis_{mode}.csv")
        os.makedirs("results", exist_ok=True)
        df_row.to_csv(csv_name, mode='a', index=False, header=not os.path.exists(csv_name))

        print(f"✅ Saved row for: {filename}")


        print(f"✅ Processed: {filename}")
        print(f"Metadata: {metadata}")
        print("=" * 60)

    df = pd.DataFrame(data, columns=columns)

    if save_csv:
        os.makedirs("results", exist_ok=True)
        csv_name = os.path.join("results", f"{os.path.basename(path)}_analysis_{mode}.csv")
        df.to_csv(csv_name, index=False)
        print(f"🎉 Saved to: {csv_name}")

    return df

if __name__ == "__main__":
    model_dir = "./models"  # wherever your .pkl files are stored
    folders = [
        "/Users/dayeon/data/UNM/RA/instagram_project/osaka",
        # "/Users/dayeon/data/UNM/RA/instagram_project/burberry"
    ]

    for folder in folders:
        process_directory(folder, model_dir=model_dir, mode='sequential')
