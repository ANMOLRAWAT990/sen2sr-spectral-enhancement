with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_c11_needle_start = '    # Visual stretching with Photographic Tone Curve & Sharpness'
old_c11_needle_end = '    rgb_gt_s = stretch_vis(hr_crop, gamma=1.45, clahe_clip=1.0, edge_boost=0.5)'

p_start = text.find(old_c11_needle_start)
p_end = text.find(old_c11_needle_end)

new_c11_stretch = """    # Visual stretching with Photographic Tone Curve & Sharpness
    def stretch_vis(arr, gamma=1.45, clahe_clip=0.0, edge_boost=0.0, shock_iters=0):
        ch = arr[:3].transpose(1, 2, 0).astype(np.float32) if (arr.ndim == 3 and arr.shape[0] >= 3) else arr.astype(np.float32)
        ch = np.nan_to_num(ch, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = safe_percentile(ch, (1.5, 98.5), default=(0.0, 1.0))
        norm = np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        corrected = np.power(norm, gamma)
        if shock_iters > 0:
            corrected = shock_filter(corrected, iterations=shock_iters, dt=0.4)
        if clahe_clip > 0 or edge_boost > 0:
            u8 = (corrected * 255.0).astype(np.uint8)
            lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            if clahe_clip > 0:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                l = clahe.apply(l)
            if edge_boost > 0:
                morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                grad = cv2.morphologyEx(l, cv2.MORPH_GRADIENT, morph_k)
                l_f = l.astype(np.float32)
                blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=0.9)
                edge = l_f - blur
                l = np.clip(l_f + edge_boost * edge + 0.25 * grad.astype(np.float32), 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return corrected

    in_8x = F.interpolate(lr_crop, scale_factor=8, mode='bicubic').squeeze(0).cpu().numpy()
    rgb_in_s = stretch_vis(in_8x, gamma=1.45, clahe_clip=0.0, edge_boost=0.0, shock_iters=0)
    rgb_s1_s = stretch_vis(s1_base_8x, gamma=1.45, clahe_clip=1.0, edge_boost=0.5, shock_iters=0)
    rgb_vhr_s = stretch_vis(vhr_calibrated, gamma=1.45, clahe_clip=1.6, edge_boost=1.5, shock_iters=3)
    rgb_gt_s = stretch_vis(hr_crop, gamma=1.45, clahe_clip=1.0, edge_boost=0.5, shock_iters=0)"""

text_updated = text[:p_start] + new_c11_stretch + text[p_end + len(old_c11_needle_end):]

with open('scratch/build_1_5meter_experiments.py', 'w', encoding='utf-8') as f:
    f.write(text_updated)

print("Updated Cell 11 in build script successfully!")

