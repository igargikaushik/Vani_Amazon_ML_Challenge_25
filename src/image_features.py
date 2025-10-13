import os 
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageStat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, EfficientNetB0
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import img_to_array
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- Configuration ---
IMAGE_DOWNLOAD_FOLDER = 'images'
IMAGE_SIZE = (224, 224)


#utilities-
def _safe_exists(path):
    return bool(path) and os.path.exists(path)

def safe_load_pil(image_path, size=IMAGE_SIZE):
    """Return numpy RGB array or None if not available."""
    if not _safe_exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert('RGB').resize(size)
        return np.array(img)
    except Exception:
        return None
def extract_color_features(image_path):
        
    """Extract color-based features that correlate with product pricing
    Returns: np.array of shape (6,) -> [mean_r, mean_g, mean_b, std_r, std_g, std_b]
    Lightweight and robust."""
    if not _safe_exists(image_path):
        return np.zeros(6, dtype=np.float32)   
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Color statistics
        stat = ImageStat.Stat(img)
        
        # Dominant colors using K-means
        img_array = np.array(img.resize((50, 50)))  # Reduce size for speed
        pixels = img_array.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Color features
       # features = features.astype(np.float32)

        features = {
            'brightness': np.mean(stat.mean),
            'contrast': np.std(stat.mean),
            'color_variance': np.var(pixels, axis=0).mean(),
            'dominant_color_1_r': dominant_colors[0][0],
            'dominant_color_1_g': dominant_colors[0][1], 
            'dominant_color_1_b': dominant_colors[0][2],
            'dominant_color_2_r': dominant_colors[1][0],
            'dominant_color_2_g': dominant_colors[1][1],
            'dominant_color_2_b': dominant_colors[1][2],
            'color_richness': len(np.unique(pixels.reshape(-1, 3), axis=0)) / len(pixels)
        }
        
        return features
        
    except Exception as e:
        return {f'brightness': 0, 'contrast': 0, 'color_variance': 0,
                'dominant_color_1_r': 0, 'dominant_color_1_g': 0, 'dominant_color_1_b': 0,
                'dominant_color_2_r': 0, 'dominant_color_2_g': 0, 'dominant_color_2_b': 0,
                'color_richness': 0}

def extract_texture_features(image_path):
    if not image_path or not os.path.exists(image_path):
     return {'texture_variance': 0, 'edge_density': 0, 'gradient_magnitude': 0}

    """Extract texture features using OpenCV"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Texture features
        features = {
            'texture_variance': np.var(gray),
            'edge_density': np.mean(cv2.Canny(gray, 50, 150)) / 255.0,
            'gradient_magnitude': np.mean(np.gradient(gray.astype(float))),
        }
        
        return features
        
    except Exception as e:
        return {'texture_variance': 0, 'edge_density': 0, 'gradient_magnitude': 0}

def extract_composition_features(image_path):
    """Extract composition and layout features"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Image composition features
        height, width = img_array.shape[:2]
        
        # Center region analysis (products often centered)
        center_h, center_w = height//4, width//4
        center_region = img_array[center_h:3*center_h, center_w:3*center_w]
        
        features = {
            'aspect_ratio': width / height,
            'center_brightness': np.mean(center_region),
            'center_vs_edge_contrast': np.mean(center_region) - np.mean(img_array),
            'image_complexity': np.std(img_array),
        }
        
        return features
        
    except Exception as e:
        return {'aspect_ratio': 1.0, 'center_brightness': 0, 'center_vs_edge_contrast': 0, 'image_complexity': 0}

def extract_packaging_features(image_path):
    """Extract features related to packaging quality and premium appearance"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Premium packaging indicators
        features = {
            'gold_presence': np.sum((hsv[:,:,0] >= 15) & (hsv[:,:,0] <= 35) & (hsv[:,:,1] > 100)) / (img_array.shape[0] * img_array.shape[1]),
            'silver_presence': np.sum((hsv[:,:,1] < 30) & (hsv[:,:,2] > 180)) / (img_array.shape[0] * img_array.shape[1]),
            'black_presence': np.sum(hsv[:,:,2] < 50) / (img_array.shape[0] * img_array.shape[1]),
            'white_presence': np.sum(hsv[:,:,2] > 200) / (img_array.shape[0] * img_array.shape[1]),
            'color_saturation': np.mean(hsv[:,:,1]),
        }
        
        return features
        
    except Exception as e:
        return {'gold_presence': 0, 'silver_presence': 0, 'black_presence': 0, 'white_presence': 0, 'color_saturation': 0}

def load_and_preprocess_image(image_path):
    """Enhanced image loading with multiple preprocessing options"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        return None

def extract_deep_features(df, model_name='resnet50', batch_size=32):
    """Extract deep features using multiple CNN architectures in batches to save memory"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available. Returning zero features.")
        return np.zeros((len(df), 2048 if model_name=='resnet50' else 1280))
    
    print(f"🔥 Initializing {model_name.upper()} for feature extraction...")
    
    # Model setup
    if model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = resnet_preprocess
        feature_size = 2048
    elif model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = efficientnet_preprocess
        feature_size = 1280
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Prepare image paths
    image_paths = [os.path.join(IMAGE_DOWNLOAD_FOLDER, os.path.basename(link))
                   if pd.notna(link) else None for link in df['image_link']]
    
    all_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Deep Features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for path in batch_paths:
            img_array = load_and_preprocess_image(path)
            if img_array is None:
                img_array = np.zeros(IMAGE_SIZE + (3,))
            batch_imgs.append(img_array)
        batch_tensor = np.array(batch_imgs)
        batch_tensor = preprocess_func(batch_tensor)
        feats = model.predict(batch_tensor, batch_size=batch_size, verbose=0)
        all_features.append(feats)
    
    features_array = np.vstack(all_features)
    print(f"✅ Deep feature extraction complete. Shape: {features_array.shape}")
    return features_array


def extract_comprehensive_image_features(df: pd.DataFrame, use_deep_features=True, model_name='resnet50'):
    """
    Extract comprehensive image features for price prediction
    """
    print("🚀 Starting Comprehensive Image Feature Extraction...")
    
    # Prepare image paths
    image_paths = []
    for link in df['image_link']:
        if pd.notna(link):
            filename = os.path.basename(link)
            image_paths.append(os.path.join(IMAGE_DOWNLOAD_FOLDER, filename))
        else:
            image_paths.append(None)
    
    # Extract traditional image features
    print("🎨 Extracting color features...")
    color_features_list = []
    for path in tqdm(image_paths, desc="Color Features"):
        if path and os.path.exists(path):
            color_features_list.append(extract_color_features(path))
        else:
            color_features_list.append(extract_color_features(None))
    
    print("🔍 Extracting texture features...")
    texture_features_list = []
    for path in tqdm(image_paths, desc="Texture Features"):
        if path and os.path.exists(path):
            texture_features_list.append(extract_texture_features(path))
        else:
            texture_features_list.append(extract_texture_features(None))
    
    print("📐 Extracting composition features...")
    composition_features_list = []
    for path in tqdm(image_paths, desc="Composition Features"):
        if path and os.path.exists(path):
            composition_features_list.append(extract_composition_features(path))
        else:
            composition_features_list.append(extract_composition_features(None))
    
    print("📦 Extracting packaging features...")
    packaging_features_list = []
    for path in tqdm(image_paths, desc="Packaging Features"):
        if path and os.path.exists(path):
            packaging_features_list.append(extract_packaging_features(path))
        else:
            packaging_features_list.append(extract_packaging_features(None))
    
    # Combine traditional features
    traditional_features = []
    for i in range(len(df)):
        combined = {**color_features_list[i], **texture_features_list[i], 
                   **composition_features_list[i], **packaging_features_list[i]}
        traditional_features.append(list(combined.values()))
    
    traditional_features = np.array(traditional_features)
    
    # Extract deep features if requested
    if use_deep_features:
        deep_features = extract_deep_features(df, model_name=model_name)
        
        # Combine traditional and deep features
        final_features = np.hstack([traditional_features, deep_features])
    else:
        final_features = traditional_features
    
    print(f"✅ Comprehensive image feature extraction complete!")
    print(f"   📏 Traditional features: {traditional_features.shape[1]}")
    if use_deep_features:
        print(f"   🧠 Deep features: {deep_features.shape[1]}")
    print(f"   🎯 Total features: {final_features.shape[1]}")
    print(f"   💾 Memory usage: {final_features.nbytes / 1024 / 1024:.1f} MB")
    
    # Create feature names for reference
    feature_names = (list(color_features_list[0].keys()) + 
                    list(texture_features_list[0].keys()) +
                    list(composition_features_list[0].keys()) +
                    list(packaging_features_list[0].keys()))
    
    if use_deep_features:
        feature_names.extend([f'{model_name}_feature_{i}' for i in range(deep_features.shape[1])])
    
    return final_features, feature_names

if __name__ == "__main__":
    print("🎯 Image Feature Extraction for Production")
    print("=" * 50)
    
    DATASET_FOLDER = 'dataset'
    TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv')
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')
    
    if not os.path.exists(IMAGE_DOWNLOAD_FOLDER):
        print(f"❌ Error: Image folder '{IMAGE_DOWNLOAD_FOLDER}' not found.")
        print("   Please download images first using data_setup.py")
        exit()
    
    # Load training data
    if os.path.exists(TRAIN_DATA_PATH):
        print(f"📂 Loading training data from: {TRAIN_DATA_PATH}")
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"   ✓ Loaded {len(train_df)} training samples")
        
        # Extract training features
        train_features, feature_names = extract_comprehensive_image_features(
            train_df, use_deep_features=True, model_name='resnet50'
        )
        print(f"   ✓ Training features shape: {train_features.shape}")
        
        # Extract test features if available
        if os.path.exists(TEST_DATA_PATH):
            print(f"📂 Loading test data from: {TEST_DATA_PATH}")
            test_df = pd.read_csv(TEST_DATA_PATH)
            print(f"   ✓ Loaded {len(test_df)} test samples")
            
            test_features, _ = extract_comprehensive_image_features(
                test_df, use_deep_features=True, model_name='resnet50'
            )
            print(f"   ✓ Test features shape: {test_features.shape}")
    else:
        print(f"❌ Error: Could not find train.csv in 'dataset/' folder")
        exit()
        
        # Extract comprehensive features
        image_features, feature_names = extract_comprehensive_image_features(
            test_df, use_deep_features=True, model_name='resnet50'
        )
        
        print("\n" + "=" * 50)
        print("📊 FEATURE ANALYSIS")
        print("=" * 50)
        print(f"Feature matrix shape: {image_features.shape}")
        print(f"Sample features (first 5 samples, first 10 features):")
        print(image_features[:5, :10])
        
        print(f"\n🏷️  Feature Names (first 20):")
        for i, name in enumerate(feature_names[:20]):
            print(f"   {i+1:2d}. {name}")
        
        print("\n✅ Enhanced image feature extraction completed!")
        print("🚀 Ready for multimodal price prediction!")
