# -*- coding: utf-8 -*-
"""
可视化识别结果
在图片上标注检测框和尺寸信息
使用 PIL 绘制 Unicode 字符（φ/×等）
"""

import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size=20):
    """获取支持 Unicode 的字体"""
    # Windows 常见字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/arial.ttf",     # Arial
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    
    # 回退到默认字体
    return ImageFont.load_default()


def draw_text_pil(img, text, pos, font, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """使用 PIL 在图片上绘制文字（支持 Unicode）
    
    Args:
        img: OpenCV 图像 (BGR)
        text: 要绘制的文字
        pos: 文字位置 (x, y)
        font: PIL 字体对象
        color: 文字颜色 (R, G, B)
        bg_color: 背景颜色 (R, G, B)
    
    Returns:
        绘制后的 OpenCV 图像
    """
    # 转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    x, y = pos
    
    # 获取文字大小
    bbox = draw.textbbox((x, y), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # 绘制背景
    padding = 3
    draw.rectangle(
        [x - padding, y - padding, x + tw + padding, y + th + padding],
        fill=bg_color
    )
    
    # 绘制文字
    draw.text((x, y), text, font=font, fill=color)
    
    # 转回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def visualize(image_path, json_path, output_path=None):
    """在图片上可视化识别结果"""
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    # 读取结果
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"图片: {image_path}")
    print(f"识别到 {len(results)} 个尺寸")

    # 颜色配置 (BGR for OpenCV)
    colors = {
        "长度": (0, 255, 0),      # 绿色
        "直径(φ)": (255, 0, 0),   # 蓝色
        "半径(R)": (0, 255, 255), # 黄色
        "螺纹(M)": (255, 0, 255), # 紫色
        "正方形(□)": (0, 140, 255), # 橙色
        "矩形(×)": (0, 165, 255), # 橙色
    }
    
    # 加载字体
    font = get_font(size=18)

    for item in results:
        idx = item["id"]
        text = item.get("text", "")  # 直接使用 JSON 中已格式化的 text 字段
        category = item["category"]
        conf = item["confidence"]
        obb_pts = item.get("obb_points", [])

        # 构建显示文本：ID + text
        display_text = f"{idx}: {text}"

        # 获取颜色
        color = colors.get(category, (0, 255, 0))

        # 画 OBB 框
        if obb_pts and len(obb_pts) == 4:
            pts = np.array(obb_pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, color, 2)

            # 文字位置（框的上方）
            min_y = min(p[1] for p in obb_pts)
            min_x = min(p[0] for p in obb_pts)
            text_x = int(min_x)
            text_y = int(min_y) - 22

            # 确保文字不超出图片
            if text_y < 5:
                text_y = int(max(p[1] for p in obb_pts)) + 5

        else:
            # 使用 rect
            rect = item["rect"]
            x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text_x = x
            text_y = y - 22 if y > 25 else y + h + 5

        # 使用 PIL 绘制文字（支持 Unicode）
        img = draw_text_pil(img, display_text, (text_x, text_y), font)

        print(f"  [{idx}] {text} ({category}) conf={conf:.2f}")

    # 保存
    if output_path is None:
        stem = Path(image_path).stem
        output_path = Path(image_path).parent.parent / "output" / f"{stem}_visualized.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), img)
    print(f"\n可视化结果已保存: {output_path}")

    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='可视化识别结果')
    parser.add_argument('--image', required=True, help='原始图片路径')
    parser.add_argument('--json', required=True, help='识别结果 JSON 文件')
    parser.add_argument('--output', help='输出图片路径')
    args = parser.parse_args()

    visualize(args.image, args.json, args.output)


if __name__ == '__main__':
    main()
