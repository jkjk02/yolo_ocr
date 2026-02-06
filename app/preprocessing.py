# -*- coding: utf-8 -*-
"""
图像预处理模块
用于处理生产环境中遇到的各种噪声干扰

支持：
- 椒盐噪声检测与去除
- 双边滤波平滑
"""

import cv2
import numpy as np


def detect_salt_pepper(img, threshold=0.01):
    """
    检测图像是否存在椒盐噪声
    
    原理：统计极端值（0 或 255）像素占比
    
    Args:
        img: 输入图像
        threshold: 极端值占比阈值（默认 1%）
    
    Returns:
        bool: 是否存在椒盐噪声
        float: 极端值占比
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    total_pixels = gray.size
    
    # 统计极端值像素
    salt_pixels = np.sum(gray == 255)
    pepper_pixels = np.sum(gray == 0)
    
    # 计算占比
    extreme_ratio = (salt_pixels + pepper_pixels) / total_pixels
    
    return extreme_ratio > threshold, extreme_ratio


def remove_salt_pepper(img, ksize=3):
    """
    去除椒盐噪声（中值滤波）
    
    Args:
        img: 输入图像
        ksize: 滤波核大小（默认 3x3）
    
    Returns:
        处理后的图像
    """
    return cv2.medianBlur(img, ksize)


def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    双边滤波（保边平滑）
    
    Args:
        img: 输入图像
        d: 滤波直径
        sigma_color: 颜色空间滤波 sigma
        sigma_space: 坐标空间滤波 sigma
    
    Returns:
        处理后的图像
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def preprocess_image(img, salt_pepper_threshold=0.01, verbose=False):
    """
    图像预处理主函数
    
    流程：
    1. 检测椒盐噪声，如有则先去除
    2. 应用双边滤波平滑
    
    Args:
        img: 输入图像
        salt_pepper_threshold: 椒盐噪声检测阈值
        verbose: 是否打印处理信息
    
    Returns:
        处理后的图像
        dict: 处理信息
    """
    info = {
        'has_salt_pepper': False,
        'salt_pepper_ratio': 0.0,
        'steps': []
    }
    
    result = img.copy()
    
    # 1. 检测并去除椒盐噪声
    has_sp, sp_ratio = detect_salt_pepper(result, salt_pepper_threshold)
    info['salt_pepper_ratio'] = sp_ratio
    
    if has_sp:
        info['has_salt_pepper'] = True
        info['steps'].append('median_blur_3x3')
        result = remove_salt_pepper(result, ksize=3)
        if verbose:
            print(f"  检测到椒盐噪声 (占比 {sp_ratio*100:.2f}%)，已应用中值滤波")
    
    # 2. 应用双边滤波
    info['steps'].append('bilateral_filter')
    result = apply_bilateral_filter(result)
    if verbose:
        print(f"  已应用双边滤波")
    
    return result, info


def create_comparison_image(original, processed, title1="原图", title2="处理后"):
    """
    创建对比图像（左右拼接）
    
    Args:
        original: 原图
        processed: 处理后的图像
        title1: 左侧标题
        title2: 右侧标题
    
    Returns:
        拼接后的图像
    """
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    # 调整到相同高度
    if h1 != h2:
        scale = h1 / h2
        processed = cv2.resize(processed, (int(w2 * scale), h1))
        h2, w2 = processed.shape[:2]
    
    # 创建拼接图像
    gap = 20
    combined = np.ones((h1 + 60, w1 + w2 + gap, 3), dtype=np.uint8) * 255
    
    # 放置图像
    combined[50:50+h1, 0:w1] = original
    combined[50:50+h2, w1+gap:w1+gap+w2] = processed
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, title1, (w1//2 - 50, 35), font, 1, (0, 0, 0), 2)
    cv2.putText(combined, title2, (w1 + gap + w2//2 - 50, 35), font, 1, (0, 0, 0), 2)
    
    # 添加分隔线
    cv2.line(combined, (w1 + gap//2, 0), (w1 + gap//2, h1 + 60), (200, 200, 200), 2)
    
    return combined
