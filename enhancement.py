import cv2
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

def get_gaussian_reference_kernels(kernel_size=0, sigma=20, channels=3, device="cuda", dtype=torch.float32):
    if kernel_size <= 0:
        kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    G_sigma_1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    G_sigma_1d = G_sigma_1d / G_sigma_1d.sum()
    G_sigma_h = G_sigma_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    G_sigma_v = G_sigma_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    return G_sigma_h, G_sigma_v

def compute_gaussian_local_reference(I0, G_sigma_h, G_sigma_v):
    c = I0.shape[1]
    pad_w = G_sigma_h.shape[-1] // 2
    pad_h = G_sigma_v.shape[-2] // 2
    L_I_c = F.pad(I0, (pad_w, pad_w, 0, 0), mode="reflect")
    L_I_c = F.conv2d(L_I_c, G_sigma_h, groups=c)
    L_I_c = F.pad(L_I_c, (0, 0, pad_h, pad_h), mode="reflect")
    L_I_c = F.conv2d(L_I_c, G_sigma_v, groups=c)
    return L_I_c

def extract_local_structural_response(I0, G_sigma_h, G_sigma_v):
    L_I_c = compute_gaussian_local_reference(I0, G_sigma_h, G_sigma_v)
    R_n = torch.log1p(I0) - torch.log1p(L_I_c)

    R_min = R_n.amin(dim=(1, 2, 3), keepdim=True)
    R_max = R_n.amax(dim=(1, 2, 3), keepdim=True)
    R_n = (R_n - R_min) / (R_max - R_min + 1e-8)
    R_n = R_n * 255.0
    return R_n

def adaptive_response_adjustment(R_n):
    b = R_n[:, 0:1, :, :]
    g = R_n[:, 1:2, :, :]
    r = R_n[:, 2:3, :, :]
    gray_R_n = 0.114 * b + 0.587 * g + 0.299 * r
    mu = gray_R_n.mean()
    eta = 1.6 - (mu / 255.0) * 0.8
    eta = torch.clamp(eta, 0.8, 1.6)
    inv_eta = 1.0 / eta
    R_n_norm = R_n / 255.0
    D = torch.pow(torch.clamp(R_n_norm, min=1e-8), inv_eta) * 255.0
    return torch.clamp(D, 0.0, 255.0)

def local_contrast_refinement(response_t, clip_limit=2.0, tile_grid_size=(8, 8)):
    response = response_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    response_lab = cv2.cvtColor(response, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(response_lab)
    contrast_operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_refined = contrast_operator.apply(L)
    refined_lab = cv2.merge((L_refined, a, b))
    refined_response = cv2.cvtColor(refined_lab, cv2.COLOR_LAB2BGR)
    return refined_response

def luminance_guided_gain_adjustment(I0, beta=0.1):
    eps = 1e-6
    avg_c = I0.mean(dim=(2, 3), keepdim=True)
    sigma_c = I0.std(dim=(2, 3), keepdim=True)
    avg = avg_c.mean(dim=1, keepdim=True)
    k_c = avg / (avg_c + beta * sigma_c + eps)
    b = I0[:, 0:1, :, :]
    g = I0[:, 1:2, :, :]
    r = I0[:, 2:3, :, :]
    gray_I0 = 0.114 * b + 0.587 * g + 0.299 * r
    omega = gray_I0 / 255.0
    G_c = torch.pow(torch.clamp(k_c, min=eps), omega)
    I_corr = I0 * G_c
    return torch.clamp(I_corr, 0.0, 255.0)

def detail_enhancement_branch(I0, G_sigma_h, G_sigma_v):
    R_n = extract_local_structural_response(I0, G_sigma_h, G_sigma_v)
    D = adaptive_response_adjustment(R_n)
    I_A = local_contrast_refinement(D, clip_limit=1.5, tile_grid_size=(4, 4))
    return I_A

def color_restoration_branch(I0):
    I_corr = luminance_guided_gain_adjustment(I0, beta=0.1)
    I_B = local_contrast_refinement(I_corr, clip_limit=2.0, tile_grid_size=(8, 8))
    return I_B

def enhance_one_image(I0_numpy, G_sigma_h, G_sigma_v, device="cuda"):
    I0 = torch.from_numpy(I0_numpy).to(device=device, dtype=torch.float32)
    I0 = I0.permute(2, 0, 1).unsqueeze(0)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    I_A = detail_enhancement_branch(I0, G_sigma_h, G_sigma_v)
    I_B = color_restoration_branch(I0)
    # fused = cv2.addWeighted(A, 0.35, B, 0.65, 0)
    I_F = cv2.addWeighted(I_A, 0.2, I_B, 0.8, 0)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    full_process_time = t1 - t0
    return I_F, full_process_time

def enhance_directory(input_dir, output_dir, sigma=40):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)
    G_sigma_h, G_sigma_v = get_gaussian_reference_kernels(
        sigma=sigma,
        channels=3,
        device=device,
        dtype=torch.float32
    )
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')
    total_full_time = 0.0
    count = 0
    warmup_done = False
    details = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(valid_exts):
            continue
        inp = os.path.join(input_dir, filename)
        out = os.path.join(output_dir, filename)
        full_start = time.perf_counter()
        I0_numpy = cv2.imread(inp)
        if I0_numpy is None:
            print("无法读取:", filename)
            continue
        h, w = I0_numpy.shape[:2]
        if device == "cuda" and not warmup_done:
            _ = enhance_one_image(I0_numpy, G_sigma_h, G_sigma_v, device=device)
            torch.cuda.synchronize()
            warmup_done = True
            full_start = time.perf_counter()
            I0_numpy = cv2.imread(inp)
            if I0_numpy is None:
                print("无法读取:", filename)
                continue
        I_F, _ = enhance_one_image(I0_numpy, G_sigma_h, G_sigma_v, device=device)
        ok = cv2.imwrite(out, I_F)
        if not ok:
            print("保存失败:", filename)
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        full_end = time.perf_counter()
        full_time = full_end - full_start
        total_full_time += full_time
        count += 1
        details.append((filename, w, h, full_time))
        print(f"处理完成: {filename} | 尺寸: {w}x{h} | 完整处理时间: {full_time:.6f} 秒")
    if count > 0:
        avg_full = total_full_time / count
        fps_full = 1.0 / avg_full if avg_full > 0 else 0.0
        print("\n========== Summary ==========")
        print(f"共处理图片数: {count}")
        print(f"\n完整总耗时: {total_full_time:.6f} 秒")
        print(f"完整平均耗时: {avg_full:.6f} 秒")
        print(f"完整FPS: {fps_full:.4f}")

        print("\n每张图片详细时间:")
        for name, w, h, full_t in details:
            print(f"{name} | 尺寸: {w}x{h} | 完整处理时间: {full_t:.6f} 秒")
    else:
        print("没有处理任何图片")

if __name__ == "__main__":
    input_dir = r"C:\Users\cly\Desktop\datasets\images\val"
    output_dir = r"C:\Users\cly\Desktop\val"
    enhance_directory(input_dir, output_dir, sigma=40)
    # "C:\Users\cly\Desktop\Paper-0001\all-images\re0.1\train"