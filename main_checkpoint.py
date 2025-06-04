import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from numba import njit
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from tensorflow.keras.losses import Loss
# Mixed Noise Injection
def add_mixed_noise(image, salt_prob=0.2, pepper_prob=0.2, gauss_mean=0, gauss_std=15):
    noisy = np.copy(image)
    total = image.shape[0] * image.shape[1]
    num_salt = int(total * salt_prob)
    num_pepper = int(total * pepper_prob)

    # Add Salt-and-Pepper noise first
    for i in range(3):
        salt_coords = [np.random.randint(0, image.shape[0], num_salt),
                       np.random.randint(0, image.shape[1], num_salt)]
        noisy[salt_coords[0], salt_coords[1], i] = 255

        pepper_coords = [np.random.randint(0, image.shape[0], num_pepper),
                         np.random.randint(0, image.shape[1], num_pepper)]
        noisy[pepper_coords[0], pepper_coords[1], i] = 0

    # Add Gaussian noise after Salt-and-Pepper
    gauss = np.random.normal(gauss_mean, gauss_std, image.shape).astype(np.float32)
    noisy = noisy.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
#  Gaussian Curvature Filter (GCF)
def apply_gcf(image, curvature_threshold, kernel_size):
    curvature = cv2.Laplacian(image, cv2.CV_64F)
    curvature = np.abs(curvature)
    mask = (((image == 0) | (image == 255)) & (curvature < curvature_threshold))
    median_filtered = cv2.medianBlur(image, kernel_size)
    output = image.copy()
    output[mask] = median_filtered[mask]
    return output

def apply_gcf_color(image, curvature_threshold, kernel_size):
    channels = cv2.split(image)
    denoised_channels = [apply_gcf(ch, curvature_threshold, kernel_size) for ch in channels]
    return cv2.merge(denoised_channels)

# MDBUTMF Filter
@njit
def apply_mdbutmf_numba(image, kernel_size):
    pad = kernel_size // 2
    height, width, channels = image.shape
    padded = np.empty((height + 2 * pad, width + 2 * pad, channels), dtype=np.uint8)
    for c in range(channels):
        for i in range(height + 2 * pad):
            for j in range(width + 2 * pad):
                ii = i - pad
                jj = j - pad
                if ii < 0:
                    ii = -ii - 1
                elif ii >= height:
                    ii = 2 * height - ii - 1
                if jj < 0:
                    jj = -jj - 1
                elif jj >= width:
                    jj = 2 * width - jj - 1
                padded[i, j, c] = image[ii, jj, c]

    output = np.copy(image)
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                if image[i, j, c] == 0 or image[i, j, c] == 255:
                    window = np.empty(kernel_size * kernel_size, dtype=np.uint8)
                    count = 0
                    for m in range(kernel_size):
                        for n in range(kernel_size):
                            window[count] = padded[i + m, j + n, c]
                            count += 1
                    valid_count = 0
                    for k in range(window.shape[0]):
                        if window[k] != 0 and window[k] != 255:
                            valid_count += 1
                    if valid_count > 0:
                        valid = np.empty(valid_count, dtype=np.uint8)
                        idx = 0
                        for k in range(window.shape[0]):
                            if window[k] != 0 and window[k] != 255:
                                valid[idx] = window[k]
                                idx += 1
                        valid.sort()
                        median = valid[valid_count // 2]
                        output[i, j, c] = median
                    else:
                        s = 0
                        for k in range(window.shape[0]):
                            s += window[k]
                        output[i, j, c] = s // window.shape[0]
    return output

def apply_mdbutmf(image, kernel_size):
    return apply_mdbutmf_numba(image, kernel_size)
# Combined loss function: Perceptual + SSIM + MSE
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, reduction='auto', name="CombinedLoss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha  # MSE weight
        self.beta = beta    # SSIM weight
        self.gamma = gamma  # Perceptual loss weight
        self.mse = MeanSquaredError()

        # Perceptual (VGG)
        vgg = VGG16(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        self.perceptual_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
        self.perceptual_model.trainable = False

    def call(self, y_true, y_pred):
        # Resize & normalize
        y_true = tf.image.resize(y_true, (224, 224))
        y_pred = tf.image.resize(y_pred, (224, 224))
        y_true_vgg = preprocess_input(y_true * 255.0)
        y_pred_vgg = preprocess_input(y_pred * 255.0)

        # MSE
        loss_mse = self.mse(y_true, y_pred)

        # SSIM (higher is better so we subtract from 1)
        loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        # Perceptual
        feat_true = self.perceptual_model(y_true_vgg)
        feat_pred = self.perceptual_model(y_pred_vgg)
        loss_percep = tf.reduce_mean(tf.square(feat_true - feat_pred))

        return self.alpha * loss_mse + self.beta * loss_ssim + self.gamma * loss_percep


# DnCNN Refinement
def apply_dncnn(image, model_path="dncnn_preatrained_mixednoise.keras"):
    image_norm = image.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(image_norm, axis=0)
    model = load_model(model_path, custom_objects={'CombinedLoss': CombinedLoss})
    denoised = model.predict(input_tensor)
    denoised = np.squeeze(denoised, axis=0)
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    return denoised
# Dynamic Window Size Adjustment 
def calculate_noise_level(image):
    salt_pixels = np.sum(image == 255)
    pepper_pixels = np.sum(image == 0)
    total_pixels = image.size
    return (salt_pixels + pepper_pixels) / total_pixels

def adjust_window_size(image, min_size=3, max_size=9):
    noise_level = calculate_noise_level(image)
    if noise_level > 0.1:  # If the noise level is high
        return max_size
    else:  # If the noise level is low
        return min_size

from skimage.exposure import match_histograms

def restore_color_tone(denoised_image, reference_image):
    # Ensure the image is in RGB format before matching
    return match_histograms(denoised_image, reference_image, channel_axis=-1)

def apply_shadow_aware_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    if np.mean(l) < 85:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        return image

def apply_light_region_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    if np.mean(l) > 180:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    else:
        return image
def apply_light_region_denoising(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    return np.mean(l) > 180

# Hybrid Denoising Pipeline
def denoise_pipeline(noisy, kernel_size, curvature_threshold, model_path, original):
    # Step 1: Region-aware enhancement (shadow + bright)
    shadow_enhanced = apply_shadow_aware_clahe(noisy)
    fully_enhanced = apply_light_region_enhancement(shadow_enhanced)

    # Step 2: Apply MDBUTMF
    mdbutmf_out = apply_mdbutmf(fully_enhanced, kernel_size)

    # Step 3: Apply GCF
    gcf_out = apply_gcf(mdbutmf_out, curvature_threshold, kernel_size)

    # Step 4: DnCNN
    dncnn_out = apply_dncnn(gcf_out, model_path)
    final_output = restore_color_tone(dncnn_out, original)
    return final_output

# Metric Functions
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return ssim(img1, img2, data_range=255, channel_axis=-1)

import cv2
import numpy as np

def calculate_gmsd(img1, img2):
    # Convert to grayscale if RGB
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    T = 170
    eps = 1e-4

    fx = np.array([[1, 0, -1]], dtype=np.float32) / 2
    fy = fx.T

    grad1_x = cv2.filter2D(img1, -1, fx)
    grad1_y = cv2.filter2D(img1, -1, fy)
    grad2_x = cv2.filter2D(img2, -1, fx)
    grad2_y = cv2.filter2D(img2, -1, fy)

    gm1 = np.sqrt(grad1_x ** 2 + grad1_y ** 2)
    gm2 = np.sqrt(grad2_x ** 2 + grad2_y ** 2)

    q_map = (2 * gm1 * gm2 + T) / (gm1 ** 2 + gm2 ** 2 + T)
    return np.sqrt(np.mean((1 - q_map) ** 2 + eps))

import torch
import lpips
from torchvision import transforms

# Initialize LPIPS model globally to avoid repeated instantiation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='vgg').to(device)
lpips_model.eval()

def calculate_lpips(img1, img2):
    # Convert from uint8 to float32 and normalize to [0, 1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Convert from numpy (H, W, C) to torch tensor (1, 3, H, W)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # LPIPS expects [-1, 1] range
    img1_tensor = (img1_tensor * 2) - 1
    img2_tensor = (img2_tensor * 2) - 1

    # Disable gradient tracking and compute LPIPS
    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor).item()

    return lpips_score

def calculate_mse(img1, img2):
    return mean_squared_error(img1.astype(np.float32).flatten(), img2.astype(np.float32).flatten())

# AMOA Optimization
def fitness_function(original, noisy, candidate_params, model_path):
    kernel_size, curvature_threshold, tolerance = candidate_params
    kernel_size = int(round(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Denoise using the current parameters
    denoised = denoise_pipeline(noisy, kernel_size, curvature_threshold, model_path, original)

    # Compute metrics
    psnr_val = calculate_psnr(original, denoised)
    ssim_val = calculate_ssim(original, denoised)
    gmsd_val = calculate_gmsd(original, denoised)
    mse_val = calculate_mse(original, denoised)
    lpips_val = calculate_lpips(original, denoised)
    # Combine into fitness score (you can adjust weights as needed)
    fitness = 0.3 * psnr_val + 0.3 * ssim_val * 100 - 0.154 * mse_val + 0.2 * (1 - gmsd_val) - 0.1 * lpips_val

    print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, GMSD: {gmsd_val:.4f}, LPIPS: {lpips_val:.4f}, MSE: {mse_val:.2f}, Fitness: {fitness:.4f}")
    return fitness

#  AMOA Optimization
def run_amoa_optimization(original, noisy, model_path, pop_size=5, iterations=5,
                          bounds=[(3,15), (5,100), (0,100)]):
    population = [[random.uniform(lb, ub) for (lb, ub) in bounds] for _ in range(pop_size)]
    velocities = [[0.0, 0.0, 0.0] for _ in range(pop_size)]
    personal_best = [p[:] for p in population]
    personal_best_scores = [fitness_function(original, noisy, p, model_path) for p in population]
    best_idx = np.argmax(personal_best_scores)
    global_best = personal_best[best_idx][:]

    for iter in range(iterations):
        for i in range(pop_size):
            r1, r2 = random.random(), random.random()
            for j in range(3):
                velocities[i][j] = (0.8 * velocities[i][j] +
                                    1.8 * r1 * (personal_best[i][j] - population[i][j]) +
                                    1.2 * r2 * (global_best[j] - population[i][j]))
                population[i][j] += velocities[i][j]
                lb, ub = bounds[j]
                population[i][j] = max(lb, min(population[i][j], ub))
            score = fitness_function(original, noisy, population[i], model_path)
            if score > personal_best_scores[i]:
                personal_best[i] = population[i][:]
                personal_best_scores[i] = score

        # Update global best
        best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[best_idx][:]
        best_score = personal_best_scores[best_idx]

        print(f"Iteration {iter}: Best Score: {best_score:.4f}")
    return global_best
# Evaluation Pipeline
def evaluate_hybrid_pipeline(input_dir, output_dir, salt_prob=0.2, pepper_prob=0.2,
                             gauss_mean=0, gauss_std=15, bounds=[(3,15), (5,100), (0,100)],
                             model_path="dncnn_pretrained_mixednoise.keras", target_size=(256,256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    psnr_list, ssim_list, mse_list, gmsd_list, lpips_list= [], [], [], [], []

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))]

    for i, filename in enumerate(tqdm(image_files, desc="Processing Images")):
        path = os.path.join(input_dir, filename)
        original = cv2.imread(path)
        if original is None:
            continue
        original = cv2.resize(original, target_size)
        # Inject mixed noise into the original image
        noisy = add_mixed_noise(original, salt_prob, pepper_prob, gauss_mean, gauss_std)

        # Optimize spatial filter parameters for this image using AMOA
        best_params = run_amoa_optimization(original, noisy, model_path, pop_size=5, iterations=5, bounds=bounds, )
        best_kernel = int(round(best_params[0]))
        if best_kernel % 2 == 0:
            best_kernel += 1
        best_curvature = best_params[1]
        best_tolerance = best_params[2]

        # Apply the hybrid pipeline with optimized parameters
        denoised = denoise_pipeline(noisy, best_kernel, best_curvature, model_path, original)
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)

        # Compute metrics
        psnr_val = calculate_psnr(original, denoised)
        ssim_val = calculate_ssim(original, denoised)
        mse_val = calculate_mse(original, denoised)
        gmsd_val = calculate_gmsd(original, denoised)
        lpips_val = calculate_lpips(original, denoised)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mse_list.append(mse_val)
        gmsd_list.append(gmsd_val)
        lpips_list.append(lpips_val)

        # Save outputs
        cv2.imwrite(os.path.join(output_dir, f"noisy_{i}.png"), noisy)
        cv2.imwrite(os.path.join(output_dir, f"denoised_{i}.png"), denoised)

    # Final average metrics
    print("\n--- Average Evaluation Metrics ---")
    print(f"Average PSNR:  {np.mean(psnr_list):.2f}")
    print(f"Average SSIM:  {np.mean(ssim_list):.4f}")
    print(f"Average MSE:   {np.mean(mse_list):.2f}")
    print(f"Average GMSD:  {np.mean(gmsd_list):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.4f}")

# Main Execution
import time
if __name__ == '__main__':
    start_time = time.time()
    input_dir = r'C:\Users\Students\Desktop\Project\WHU-RS19 -main\WHU-RS19\new_dataset_WHU_RS'  # dataset path
    output_dir = r'C:\Users\Students\Desktop\Project\WHU-RS19 -main\output\0.2'  # output path
    evaluate_hybrid_pipeline(input_dir, output_dir,
                             salt_prob=0.2, pepper_prob=0.2,
                             gauss_mean=0, gauss_std=15,
                             bounds=[(3,15), (5,100), (0,100)],
                             model_path="dncnn_pretrained_mixednoise.keras",
                             target_size=(256,256))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n Total Time Taken: {total_time:.2f} seconds")

