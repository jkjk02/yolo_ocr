# -*- coding: utf-8 -*-
"""
YOLO OCR 尺寸识别器 - 优化版 v2
作用：从工程图纸中自动识别尺寸标注

核心优化：
1. 扩大 ROI 区域识别（解决公差截断问题）
2. 图像预处理（椒盐噪声检测 + 双边滤波）
3. 符号检测（model_b 检测 φ/R/M）
4. 优化归并逻辑（主值 +上公差 -下公差）
5. 加强过滤（标题栏、序号、人名等）
"""

# 压制警告（包括 CPU ExecutionProvider 警告）
import os
os.environ['ONNXRUNTIME_LOG_SEVERITY_LEVEL'] = '4'  # FATAL only
os.environ['ORT_TENSORRT_UNAVAILABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 禁用 ONNX Runtime 的所有警告输出（C++ 级别）
os.environ['ORT_DISABLE_WARNINGS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只用一个 GPU

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*CUDAExecutionProvider.*')
warnings.filterwarnings('ignore', message='.*Fallback.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import re
import json
import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np

# 添加项目路径
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from app.preprocessing import preprocess_image
from app.visualizer import visualize  # 集成可视化

# ============== 路径配置 (适配 PyInstaller) ==============
def get_base_path():
    """获取基础路径，兼容源码运行和 PyInstaller 打包运行"""
    if getattr(sys, 'frozen', False):
        # 打包模式：使用解压后的临时目录
        return Path(sys._MEIPASS)
    else:
        # 源码模式：使用当前文件上两级目录 (app/../ -> root)
        return Path(__file__).parent.parent

BASE_PATH = get_base_path()

# 模型路径配置
MODEL_A_PATH = str(BASE_PATH / "best-v3.onnx")
MODEL_B_PATH = str(BASE_PATH / "model_b" / "weights" / "best.onnx")

# OCR 模型配置
OCR_DET_MODEL = str(BASE_PATH / "onnxocr/models/ch_ppocr_server_v2.0/det/det.onnx")
OCR_REC_MODEL = str(BASE_PATH / "onnxocr/models/ppocrv5/rec/rec.onnx")
OCR_DICT_PATH = str(BASE_PATH / "onnxocr/models/ppocrv5/ppocrv5_dict.txt")

# 符号类型映射
SYMBOL_NAMES = {0: "diameter", 1: "radius", 2: "thread"}
SYMBOL_PREFIX = {"diameter": "φ", "radius": "R", "thread": "M", "square": "□"}

# ============== 功能开关 ==============
# 调试开关：如果在某些图片上效果变差，尝试关闭这些选项
ENABLE_SYMBOL_ERASE = False   # 是否在 OCR 前擦除符号（可能误伤数字）
ENABLE_ROI_EXPAND = True      # 是否扩大 OCR 识别区域（为了包含公差）
ENABLE_PREPROCESSING = False  # 是否进行去噪预处理（可能把小字抹掉）
ENABLE_PERSPECTIVE_CROP = False # 是否使用透视变换裁剪（False=简单矩形裁剪）
ENABLE_DEBUG_SAVE_ROI = True  # 是否保存每个 Box 的裁剪图片用于调试



# ============== 工具函数 ==============

def expand_roi(img, obb_pts, expand_ratio=0.5, expand_ratio_x=None):
    """扩大 ROI 区域（解决公差截断问题）- 已弃用，使用 crop_obb 代替
    
    Args:
        img: 原始图像
        obb_pts: OBB 四个角点
        expand_ratio: 垂直方向扩展比例
        expand_ratio_x: 水平方向扩展比例（默认 = expand_ratio * 1.5）
    """
    pts = np.array(obb_pts, dtype=np.float32)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # 水平方向扩展更多（公差通常在右侧）
    if expand_ratio_x is None:
        expand_ratio_x = expand_ratio * 1.5
    
    roi_x1 = max(0, int(x_min - width * expand_ratio_x))
    roi_y1 = max(0, int(y_min - height * expand_ratio))
    roi_x2 = min(img.shape[1], int(x_max + width * expand_ratio_x))
    roi_y2 = min(img.shape[0], int(y_max + height * expand_ratio))
    
    if roi_x2 - roi_x1 < 10 or roi_y2 - roi_y1 < 10:
        return None
    
    return img[roi_y1:roi_y2, roi_x1:roi_x2].copy()


def crop_obb(img, obb_pts, scale=1.3):
    """透视变换裁剪 OBB 区域（推荐方法）"""
    pts = np.array(obb_pts, dtype=np.float32)

    # 计算宽高
    # pts顺序: 0-1是宽(Top), 1-2是高(Right)
    width = np.linalg.norm(pts[0] - pts[1])
    height = np.linalg.norm(pts[1] - pts[2])

    # 扩大范围逻辑优化
    # 如果是细长矩形 (Aspect Ratio > 1.5)，说明可能是竖排文字或长条标注
    # 此时在长边方向给予更大的扩展比例，以覆盖可能遗漏的前后缀 (如 4-...)
    aspect_ratio = max(width, height) / max(min(width, height), 1)

    scale_w = scale
    scale_h = scale

    if aspect_ratio > 1.15:
        # 细长框，长边方向多扩
        if height > width:
            scale_h = scale * 1.8 # 竖排：高度方向强力扩展 (1.5 -> 1.8)
            scale_w = scale * 1.2 # 宽度方向微调 (1.1 -> 1.2)
        else:
            scale_w = scale * 1.8 # 横排：宽度方向强力扩展
            scale_h = scale * 1.2 # 高度方向常规扩展
    else:
        # 普通框：高度方向常规扩展 (照顾上下标)
        scale_h = scale * 1.2

    # 变换后的宽和高
    target_width = width * scale_w
    target_height = height * scale_h

    # 构建目标点 (从左上角顺时针)
    # 0,0 -> w,0 -> w,h -> 0,h
    dst_pts = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)

    # 获取变换矩阵
    src_pts = pts
    w = int(width)
    h = int(height)

    # 这里的 dst_pts_orig 必须对应上面的 dst_pts 的比例逻辑
    # 我们希望将 src_pts 映射到 target_size 的中心
    # 最简单的方法是直接映射到 target_width/height 矩形，不用中间 padding

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (int(target_width), int(target_height)))

    # 旋转校正：如果高 > 宽，旋转 90 度
    if warped.shape[0] > warped.shape[1] * 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return warped, "Perspective+Rot90"

    return warped, "Perspective"

def crop_box_simple(img, obb_pts, padding=30):
    """简单外接矩形裁剪（不进行透视变换）
    有时候透视变换会因为 OBB 点不准导致图像扭曲，简单裁剪反而更稳

    Args:
        padding: 扩展像素（增大到20，确保不截断）
    """
    pts = np.array(obb_pts, dtype=np.int32)
    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])
    x_max = np.max(pts[:, 0])
    y_max = np.max(pts[:, 1])
    
    # 扩展
    h, w = img.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    if x_max - x_min < 5 or y_max - y_min < 5:
        return None, "Too small"
        
    crop = img[y_min:y_max, x_min:x_max].copy()
    
    # 恢复手动旋转：PaddleOCR的方向分类器在工程图纸上表现不佳
    # 如果裁剪出的图像高大于宽，说明可能是竖排文字
    h, w = crop.shape[:2]
    rotation_info = "Simple"

    # 旋转条件：高度 > 宽度 * 1.1
    # 降低阈值到1.1，可以捕获接近正方形的竖排文字（如Box 26: 85x75, 比例1.13）
    if h > w * 1.1:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotation_info = "Simple+Rot90"

    return crop, rotation_info


def expand_image(img, scale=2.5, min_height=64):
    """放大图像用于 OCR"""
    h, w = img.shape[:2]
    if h < min_height:
        scale = max(scale, min_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LANCZOS4)


def correct_6_to_9(text, img):
    """修正 6/9 混淆问题

    策略：基于上下文和图像特征判断
    1. 如果主值是 "66" 且下公差是 "-6"，很可能是 "99" 和 "-0" 被误识别
    2. 如果只有单个 "6"，分析图像特征判断

    注意：只对纯数值类文本进行修正，避免修改材料标注等
    """
    if '6' not in text:
        return text

    # 检查是否是纯数值类文本（只包含数字、小数点、空格、正负号）
    # 如果包含字母（除了公差符号），则不修正
    import re
    text_check = text.replace('+', '').replace('-', '').replace('.', '').replace(' ', '')
    if not text_check.replace('6', '').replace('9', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('7', '').replace('8', '').isdigit() and text_check:
        # 包含非数字字符，检查是否只是数字
        if not all(c.isdigit() or c in '.+-± ' for c in text):
            return text

    # 策略1：上下文判断 - "66 ... -6" 模式很可能是 "99 ... -0"
    # 因为工程图中 "99 +0.1/-0" 这种格式很常见，而 "66 +0.1/-6" 不太合理
    pattern = r'^66\s*[+\-].*-6$'
    if re.match(pattern, text):
        # 将所有 6 替换为 9，将末尾的 -6 替换为 -0
        text = text.replace('66', '99')
        if text.endswith('-6'):
            text = text[:-1] + '0'
        return text

    # 策略2：如果主值是两位相同数字 "66"，检查是否应该是 "99"
    # 通过分析图像顶部和底部的像素分布
    if '66' in text and img is not None:
        try:
            # 转灰度并二值化
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

            h, w = binary.shape

            # 分析顶部和底部区域
            top_region = binary[:int(h * 0.3), :]
            bottom_region = binary[int(h * 0.7):, :]

            # 计算顶部和底部的像素密度
            top_density = np.sum(top_region) / (top_region.shape[0] * top_region.shape[1])
            bottom_density = np.sum(bottom_region) / (bottom_region.shape[0] * bottom_region.shape[1])

            # 如果底部密度明显低于顶部，说明底部是直线（"9"的特征）
            # "9" 的圆圈在上方，底部是直线，所以顶部密度高，底部密度低
            # "6" 的圆圈在下方，顶部是直线，所以顶部密度低，底部密度高
            if top_density > bottom_density * 1.2:
                text = text.replace('66', '99')
                # 同时修正可能的 -6 -> -0
                if '-6' in text and '+' in text:
                    text = text.replace('-6', '-0')
        except Exception:
            pass

    return text


def parse_text_components(text):
    """解析 OCR 文本，分离主值和嵌入的公差"""
    text = text.strip()
    if not text:
        return None, []
    
    # 纯公差文本
    if text.startswith('+') or text.startswith('-'):
        return None, [text]
    
    # 匹配: 主值 + 嵌入公差
    pattern = r'^([φΦRrMm□]?\d+\.?\d*)([-+]\d*\.?\d*)$'
    match = re.match(pattern, text)
    if match:
        main_part = match.group(1)
        tol_part = match.group(2)
        if tol_part:
            return main_part, [tol_part]
        return main_part, []
    
    return text, []


def merge_ocr_results(ocr_result):
    """归并 OCR 结果：主值 +上公差 -下公差"""
    if not ocr_result or not ocr_result[0]:
        return "", 0
    
    items = ocr_result[0]
    main_candidates = []
    all_tolerances = []
    confs = []
    
    for item in items:
        text = item[1][0].strip()
        conf = item[1][1]
        confs.append(conf)
        
        main_part, tol_parts = parse_text_components(text)
        if main_part:
            main_candidates.append((main_part, conf))
        for tol in tol_parts:
            all_tolerances.append((tol, conf))
    
    if not main_candidates:
        # 如果只有公差，尝试返回最大的公差作为主值（或者是纯文本）
        # 这通常发生在 OCR 把主值误识别为公差，或者只检测到了公差部分
        if items:
            # 简单地把所有识别到的文本拼起来返回
            full_text = " ".join([item[1][0] for item in items])
            avg_conf = sum(confs) / len(confs) if confs else 0
            return full_text, avg_conf
        return "", 0
    
    # 选主值（数值最大的）
    def extract_num(t):
        nums = re.findall(r'\d+\.?\d*', t)
        return float(nums[0]) if nums else 0
    
    main_candidates.sort(key=lambda x: extract_num(x[0]), reverse=True)
    main_value = main_candidates[0][0]
    
    # 分类公差
    upper_tol = lower_tol = None
    for tol_text, _ in all_tolerances:
        if tol_text.startswith('+') and not upper_tol:
            upper_tol = tol_text
        elif tol_text.startswith('-') and not lower_tol:
            lower_tol = tol_text
    
    # 组合结果
    result = main_value
    if upper_tol:
        result += f" {upper_tol}"
    if lower_tol:
        result += f" {lower_tol}"
    
    return result, sum(confs) / len(confs) if confs else 0


def parse_dimension_text(text, symbol_type=None):
    """解析尺寸文本，提取主值、公差和数量
    
    处理场景：
    - φ0.5 → 主值 0.5（φ 由 model_b 检测）
    - 2-φ2.5-0.1 → 数量 2，主值 2.5，下公差 -0.1
    - 2-02.5-0.1 → OCR 误识别，等价于 2-φ2.5-0.1
    - 14.05×12.05 → 主值 14.05，次值 12.05（矩形尺寸）
    - 2-M1.4深2.5 → 数量 2，主值 1.4，深度 2.5（螺纹孔深度）
    """
    result = {
        "main_value": None,
        "secondary_value": None,  # × 后面的第二个值
        "depth": None,            # 新增：深度值
        "quantity": None,
        "tolerance": {"type": "none", "upper": None, "lower": None},
        "dimension_type": symbol_type or "length",
        "raw_text": text
    }
    
    if not text:
        return result
    
    text = text.strip()
    
    # 移除括号
    text = text.replace('（', '').replace('）', '').replace('(', '').replace(')', '')

    # 符号替换（但不包括 '0'，因为 0.X 是有效数字）
    # 新增：正方形符号替换 (OCR 常把 □ 识别为 '口' 或 '0')
    if '口' in text:
        text = text.replace('口', '□')

    phi_chars = ['桅', '鈲', '?', '须', '蠁', '⌀', 'Ф', 'Φ', 'ф', '¢']
    for c in phi_chars:
        text = text.replace(c, 'φ')

    # 0. 修复粘连格式（如 18.50.1 -> 18.5 -0.1）
    # 模式：数字 + 小数点 + 数字 + 0.X (0.1, 0.05 等)
    # 例如：18.50.1 -> 18.5 -0.1
    # 工程图中 "+0/-0.1" 格式很常见，OCR 容易漏掉 "+0/-"，只剩下 "18.50.1"
    # 所以默认当作下公差 -0.1
    sticky_match = re.search(r'(\d+\.\d+)(0\.\d+)$', text)
    if sticky_match:
        main_val = sticky_match.group(1)
        tol_val = sticky_match.group(2)
        # 只有当公差部分看起来合理（<1）时才拆分
        if float(tol_val) < 1.0:
            # 默认当作下公差（工程图中 +0/-0.1 格式更常见）
            text = f"{main_val} +0 -{tol_val}"
    
    # 检测 A×B 格式（矩形尺寸，如 14.05×12.05）
    # 注意：只匹配浮点数×浮点数格式，排除 2-φ4.5 这种数量格式
    rect_match = re.match(r'^(\d+\.?\d*)\s*[×xX]\s*(\d+\.?\d*)(.*)$', text)
    if rect_match:
        try:
            val1 = float(rect_match.group(1))
            val2 = float(rect_match.group(2))
            # 只有当两个值都大于 1 时才认为是矩形尺寸（排除 2-φ4.5）
            if val1 > 1 and val2 > 1:
                result["main_value"] = val1
                result["secondary_value"] = val2
                result["dimension_type"] = "rectangle"
                # 继续解析剩余部分的公差
                remaining = rect_match.group(3).strip()
                if remaining:
                    # 解析公差
                    pm_match = re.search(r'±\s*(\d+\.?\d*)', remaining)
                    if pm_match:
                        tol = float(pm_match.group(1))
                        result["tolerance"] = {"type": "symmetric", "upper": tol, "lower": -tol}
                    else:
                        upper_match = re.search(r'\+\s*(\d+\.?\d*)', remaining)
                        lower_match = re.search(r'-\s*(\d+\.?\d*)', remaining)
                        if upper_match:
                            result["tolerance"]["upper"] = float(upper_match.group(1))
                            result["tolerance"]["type"] = "asymmetric"
                        if lower_match:
                            result["tolerance"]["lower"] = -float(lower_match.group(1))
                            result["tolerance"]["type"] = "asymmetric"
                return result
        except:
            pass
    
    # 解析数量-尺寸格式 (如: 2-φ4.5, 3-M6, 4-12.5, 2-02.5-0.1)
    # 注意：要区分 "6-0.03"（主值6，公差-0.03）和 "2-φ4.5"（数量2，尺寸φ4.5）
    # 规则：只有当后面是符号开头(φ/R/M/0开头的多位数)时才认为是数量格式
    qty_match = re.match(r'^(\d+)\s*[-xX×]\s*(.+)$', text)
    if qty_match:
        qty_str = qty_match.group(1)
        qty = int(qty_str)
        remaining = qty_match.group(2).strip()
        
        # 判断是否真的是数量格式
        is_quantity_format = False
        if 1 <= qty <= 20:
            # 情况1: 后面以符号开头 (φ, Φ, R, M, 或 0+数字如02.5)
            if re.match(r'^[φΦRrMm]', remaining):
                is_quantity_format = True
            # 情况2: 后面是 0+数字 格式 (如 02.5，表示 φ 被误识别为 0)
            elif re.match(r'^0\d', remaining):
                is_quantity_format = True
            # 情况3: 后面是多位整数 (如 12, 表示 2-12 = 2个12mm孔)
            elif re.match(r'^\d{2,}', remaining):
                is_quantity_format = True
            # 情况4: model_b 已经检测到符号类型（φ/R/M），说明这确实是数量格式
            # 例如 "2- 2.5-0.1" 中 φ 被擦除后留下空格
            elif symbol_type in ("diameter", "radius", "thread"):
                is_quantity_format = True
        
        if is_quantity_format:
            result["quantity"] = qty
            text = remaining
    
    # 设置维度类型
    # 优先信任 OCR 文本中的明确符号
    if text.upper().startswith('M') and re.match(r'^M\d', text.upper()):
        result["dimension_type"] = "thread"
    elif text.upper().startswith('R') and not text.upper().startswith('RA'):
        result["dimension_type"] = "radius"
    elif 'φ' in text:
        result["dimension_type"] = "diameter"
    elif '□' in text:
        result["dimension_type"] = "square"
    elif symbol_type:
        # 如果 OCR 没识别出符号，但检测模型识别出了，使用检测结果
        result["dimension_type"] = symbol_type

        # 针对检测到符号的情况，预处理文本以修复常见 OCR 错误
        # 1. 修复 "02.5" -> "2.5" (0 被误识别为 φ)
        if re.match(r'^0\d', text):
            text = text[1:]
        # 2. 修复 "-01.6" -> "1.6" (φ 被误识别为 -)
        elif re.match(r'^-\s*0?\d', text):
            text = re.sub(r'^-\s*0?', '', text)

    # 提取深度值（如: M1.4深2.5, φ3深5）
    depth_match = re.search(r'深\s*(\d+\.?\d*)', text)
    if depth_match:
        result["depth"] = float(depth_match.group(1))
        # 从文本中移除深度部分，以便后续正确解析主值
        text = text[:depth_match.start()] + text[depth_match.end():]
    
    # 提取公差 ±（对称公差）
    pm_match = re.search(r'±\s*(\d+\.?\d*)', text)
    if pm_match:
        val = float(pm_match.group(1))
        result["tolerance"] = {"type": "symmetric", "upper": val, "lower": -val}
        # 从文本中移除公差部分，以便后续提取主值
        text = text[:pm_match.start()] + text[pm_match.end():]
    else:
        # 非对称公差：+X 和 -X
        # 先找 +X 格式的上公差
        upper_match = re.search(r'\+\s*(\d+\.?\d*)', text)
        if upper_match:
            val = float(upper_match.group(1))
            result["tolerance"]["upper"] = val
            result["tolerance"]["type"] = "asymmetric"
        
        # 找 -X 格式的下公差
        # 对于带符号类型（φ/R/M）的情况，格式通常是 "主值-公差"，如 "2.5-0.1"
        # 对于普通长度，需要排除 "6-0.03" 这种主值-公差格式（前面数量解析已处理）
        if symbol_type in ("diameter", "radius", "thread"):
            # 带符号类型：直接匹配 -数字 格式（如 2.5-0.1 中的 -0.1）
            lower_match = re.search(r'-\s*(\d+\.?\d*)', text)
            if lower_match:
                val = float(lower_match.group(1))
                result["tolerance"]["lower"] = -val
                result["tolerance"]["type"] = "asymmetric"
        else:
            # 普通长度：只匹配独立的 -数字 格式（前面不是数字或小数点）
            lower_matches = list(re.finditer(r'(?<![0-9.])-\s*(\d+\.?\d*)', text))
            if lower_matches:
                last_match = lower_matches[-1]
                val = float(last_match.group(1))
                result["tolerance"]["lower"] = -val
                result["tolerance"]["type"] = "asymmetric"
    
    # 提取主值
    # 1. 尝试匹配符号后的数值
    main_patterns = [
        r'[φΦ]\s*(\d+\.?\d*)',
        r'^[Rr]\s*(\d+\.?\d*)',
        r'^[Mm]\s*(\d+\.?\d*)',
    ]
    
    for pattern in main_patterns:
        m = re.search(pattern, text)
        if m:
            result["main_value"] = float(m.group(1))
            return result
    
    # 2. 如果 model_b 检测到 φ/R/M，但 OCR 没识别出符号
    # 处理 "02.5-0.1" 这种情况：0 是 φ 的误识别
    if symbol_type:
        # 情况A: 0开头 (02.5 -> 2.5)
        m = re.match(r'^0(\d+\.?\d*)', text)
        if m:
            result["main_value"] = float(m.group(1))
            return result

        # 情况B: 负号开头，但有符号检测 (-01.6 -> 1.6)
        # 这通常是 φ 被误识别为 - 或 -0
        m = re.match(r'^-\s*0?(\d+\.?\d*)', text)
        if m:
            result["main_value"] = float(m.group(1))
            return result

        # 匹配纯数字（如 0.5）
        m = re.match(r'^(\d+\.?\d*)', text)
        if m:
            result["main_value"] = float(m.group(1))
            return result

    # 3. 通用提取：找第一个数值（排除公差中的值）
    # 移除公差部分后提取
    text_for_main = re.sub(r'[+-]\s*\d+\.?\d*', '', text)
    nums = re.findall(r'(\d+\.?\d*)', text_for_main)
    valid_nums = []
    for n in nums:
        try:
            val = float(n)
            if val > 0:
                valid_nums.append(val)
        except:
            pass
    
    if valid_nums:
        # 取第一个有效数值（通常是主值）
        result["main_value"] = valid_nums[0]

    # ========== 公差合理性检查 ==========
    # 修复 "-0" 被误识别为 "-9" 等问题
    # 工程图纸中，公差通常是小数值（如 ±0.05, +0.03/-0）
    # 如果下公差是整数且绝对值 >= 1，而上公差是小数，很可能是误识别
    tol = result.get("tolerance", {})
    upper = tol.get("upper")
    lower = tol.get("lower")

    if upper is not None and lower is not None:
        # 情况1: 上公差是小数（如 0.03），下公差是整数（如 -9）
        # 很可能是 "-0" 被误识别为 "-9"
        if 0 < upper < 1 and lower is not None and lower == int(lower) and abs(lower) >= 1:
            # 将下公差修正为 0
            result["tolerance"]["lower"] = 0

        # 情况2: 下公差是小数（如 -0.05），上公差是整数（如 9）
        # 很可能是 "+0" 被误识别为 "+9"
        if upper is not None and upper == int(upper) and abs(upper) >= 1 and lower is not None and -1 < lower < 0:
            # 将上公差修正为 0
            result["tolerance"]["upper"] = 0

    return result


# ============== 过滤器 ==============

class DimensionFilter:
    """增强过滤器"""
    
    # 标题栏关键词
    TITLE_KEYWORDS = [
        '日期', '签字', '比例', '设计', '审核', '材料', '图号', '版本',
        'DATE', 'SCALE', '单位', '共', '第', '标记', '更改', '签名',
        '质量', '工艺', '技术', '年', '月', '说明', 'PA', 'GF', '3D',
        '斯兴华', '档案', '底图', '审', '字', '期', '名称', '图纸',
        '重量', '批准', '标准化', '会签', '处数', '更改内容'
    ]
    
    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w
    
    def filter(self, dimension, box_rect):
        """执行过滤，返回 (should_keep, reason)"""
        main_value = dimension.get("main_value")
        raw_text = dimension.get("raw_text", "")
        
        # 1. 无主值
        if main_value is None:
            return False, "ValueIsNone"
        
        # 2. 数值为 0
        if main_value == 0:
            return False, "ValueIsZero"
        
        # 3. 数值过大
        if main_value > 500:
            return False, f"ValueTooLarge ({main_value})"
        
        # 4. 数字位数过多（乱码）
        main_str = str(main_value).replace('.', '')
        if len(main_str) > 6:
            return False, f"TooManyDigits ({main_str})"
        
        # 5. 标题栏关键词
        for kw in self.TITLE_KEYWORDS:
            if kw in raw_text:
                return False, f"TitleKeyword ({kw})"
        
        # 6. 中文文本过滤（增强版）
        # 6a. 纯中文文本检测：移除数字和常见符号后，如果剩余全是中文，过滤掉
        text_no_digits = re.sub(r'[\d\.\-\+±×xX\s\(\)\（\）]', '', raw_text)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_no_digits)
        non_chinese = re.sub(r'[\u4e00-\u9fff]', '', text_no_digits)
        # 如果中文字符 >= 2 且非中文部分很少，认为是纯中文标注
        if len(chinese_chars) >= 2 and len(non_chinese) <= 1:
            return False, f"ChineseText ({raw_text})"
        
        # 6b. 常见误识别中文词汇
        chinese_keywords = ['导光', '通孔', '沉孔', '倒角', '螺纹', '表面', '粗糙', 
                           '光洁', '热处理', '淬火', '镀锌', '喷漆', '抛光', '去毛刺',
                           '不锈钢', '铝合金', '碳钢', '铸铁', '黄铜', '紫铜']
        for kw in chinese_keywords:
            if kw in raw_text:
                return False, f"ChineseKeyword ({kw})"
        
        # 6c. 过多字母/中文（原逻辑，阈值调整）
        text_check = raw_text.upper()
        for c in ['M', 'R', 'X', 'Φ', 'φ', '±', '深']:
            text_check = text_check.replace(c, '')
        letter_count = len(re.findall(r'[A-Z\u4e00-\u9fff]', text_check))
        if letter_count >= 3:  # 从 >3 改为 >=3
            return False, f"TooManyLetters ({letter_count})"
        
        # 7. 简单序号（1-10 的纯整数）
        # 但如果检测框是扁平矩形（长宽比 > 2），说明可能是真实尺寸标注
        if main_value == int(main_value) and 1 <= main_value <= 10:
            digits = re.findall(r'\\d+', raw_text)
            has_symbol = any(s in raw_text for s in ['φ', 'Φ', 'R', 'M', '±', '+', '-', '.', '深'])
            if len(digits) == 1 and not has_symbol:
                # 检查长宽比：尺寸标注通常是扁平的（有尺寸线）
                box_w, box_h = box_rect[2], box_rect[3]
                aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)
                # 如果长宽比 > 2.5，可能是真实尺寸，不过滤
                if aspect_ratio <= 2.5:
                    return False, f"SimpleNumber ({int(main_value)})"
        
        # 8. 比例标注（如 2:1, 3.3:1）
        # 修复：30.4 误识别为 30:4 时不应过滤
        raw_text_normalized = raw_text.replace('：', ':')
        if ':' in raw_text_normalized:
            # 只有明确包含关键词，或者是常见的整数比例 (1:X, X:1, 2:1, 5:1) 才过滤
            is_keyword = any(k in raw_text_normalized.upper() for k in ['SCALE', '比例'])
            match = re.search(r'^(\d+):(\d+)$', raw_text_normalized)
            
            if is_keyword:
                return False, "ScaleAnnotation"
            
            if match:
                v1, v2 = int(match.group(1)), int(match.group(2))
                # 常见比例通常包含 1, 2, 5, 10，且数值较小
                if (v1 in [1, 2, 5, 10] or v2 in [1, 2, 5, 10]) and v1 < 100 and v2 < 100:
                    return False, f"ScaleAnnotation ({v1}:{v2})"
        
        # 9. 剖面标记（如 A-A, B-B, 或误识别的 8-8, 1-1）
        section_text = raw_text.strip().upper()
        # 匹配 X-X 格式（相同字符重复）
        section_match = re.match(r'^([A-Z0-9])-\1$', section_text)
        if section_match:
            return False, f"SectionMark ({raw_text})"
        # 也匹配带空格的情况：A - A
        section_match2 = re.match(r'^([A-Z0-9])\s*-\s*\1$', section_text)
        if section_match2:
            return False, f"SectionMark ({raw_text})"

        # 10. 误识别为 100/0 的视图标签 (针对 B -> 100 问题)
        # 特征：主值为 100 或 0，且没有公差，没有数量，且边框接近正方形
        if main_value in [100, 0, 8, 10]:
            has_tol = dimension.get("tolerance", {}).get("type") != "none"
            has_qty = dimension.get("quantity")
            if not has_tol and not has_qty:
                box_w, box_h = box_rect[2], box_rect[3]
                # 计算长宽比 (长边/短边)
                aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)
                # 视图标签通常是方形的 (比率接近 1.0)，而尺寸通常是长条形的 (比率 > 2.0)
                if aspect_ratio < 1.6:
                    return False, f"ViewLabelConfusion ({main_value}, ratio={aspect_ratio:.1f})"

        return True, None


# ============== 主识别器类 ==============

class DimensionRecognizer:
    """尺寸识别器 v2"""
    
    def __init__(self, model_a_path=None, model_b_path=None, use_gpu=True):
        print("加载模型...")
        
        # 压制 ultralytics 的日志输出
        import logging
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        from ultralytics import YOLO
        self.model_a = YOLO(model_a_path or MODEL_A_PATH, task='obb')
        self.model_b = YOLO(model_b_path or MODEL_B_PATH, task='detect')
        
        # 设置设备：use_gpu=False 时全部用 CPU
        self.model_a_device = 'cuda:0' if use_gpu else 'cpu'
        self.model_b_device = 'cpu'  # 符号检测始终用 CPU（ROI 小，CPU 足够快）
        
        # 加载 OCR（hybrid 组合）
        self.ocr = ONNXPaddleOcr(
            use_angle_cls=True,
            use_gpu=use_gpu,
            det_model_dir=OCR_DET_MODEL,
            rec_model_dir=OCR_REC_MODEL,
            rec_char_dict_path=OCR_DICT_PATH,
            det_db_unclip_ratio=3.5,  # 从2.0增大到3.5，扩大文本检测区域
            det_limit_side_len=3000   # 从2000增大到3000，处理更大图片
        )
        device_str = "GPU" if use_gpu else "CPU"
        print(f"模型加载完成 (使用 {device_str})")
    
    def _is_text_truncated(self, text):
        """检查文本是否可能被截断"""
        if not text:
            return False
        text = text.strip()
        # 1. 以连接符或小数点结尾
        if text[-1] in ['.', '+', '-', '×', 'x', 'X', '±', ':', '：']:
            return True
        # 2. 以 +0 或 -0 结尾
        if re.search(r'[-+±]0$', text):
            return True
        # 3. 只有前半个括号
        if ('(' in text or '（' in text) and (')' not in text and '）' not in text):
            return True
        # 4. 数字以 0 结尾且前面是符号 (如 -0.)
        if re.search(r'[-+±]0\.$', text):
            return True

        # 5. 开头疑似截断 (新增)
        # 以连字符开头 (如 "-φ1.6" 可能是 "4-φ1.6" 的截断)
        if text.startswith('-'):
            return True
        # 以 0 开头且后面紧跟数字 (如 "01.6" 可能是 "101.6" 或符号截断)
        # 但要排除 0.X 的情况
        if text.startswith('0') and len(text) > 1 and text[1] != '.':
            return True

        return False

    def _get_smart_crop(self, img, pts, scale=1.0):
        """智能裁剪：根据角度选择裁剪方式"""
        width = np.linalg.norm(pts[0] - pts[1])
        height = np.linalg.norm(pts[1] - pts[2])
        if width > height:
            angle = np.degrees(np.arctan2(pts[1,1] - pts[0,1], pts[1,0] - pts[0,0]))
        else:
            angle = np.degrees(np.arctan2(pts[2,1] - pts[1,1], pts[2,0] - pts[1,0]))
        
        angle = abs(angle % 180)
        is_orthogonal = (angle < 5 or angle > 175) or (85 < angle < 95)
        
        use_perspective = ENABLE_PERSPECTIVE_CROP and not is_orthogonal
        
        if use_perspective:
            return crop_obb(img, pts, scale=scale)
        else:
            # 简单裁剪 padding 策略
            # scale=1.0 -> padding=2
            # scale=1.3 -> padding=17
            # scale=1.6 -> padding=32
            padding = 2 + int((scale - 1.0) * 50)
            return crop_box_simple(img, pts, padding=padding)

    def process_image(self, image_path, conf=0.1, verbose=True):
        """处理图像，返回识别结果列表"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []

        h, w = img.shape[:2]
        if verbose:
            print(f"图片尺寸: {w}x{h}")

        # 创建调试输出目录
        debug_dir = None
        if ENABLE_DEBUG_SAVE_ROI:
            debug_dir = Path(image_path).parent.parent / "debug_roi"
            debug_dir.mkdir(exist_ok=True)
            if verbose:
                print(f"调试图片将保存到: {debug_dir}")
        
        # 预处理
        if ENABLE_PREPROCESSING:
            img, preprocess_info = preprocess_image(img, verbose=verbose)
        else:
            if verbose:
                print("预处理已禁用")
        
        # 创建过滤器
        dim_filter = DimensionFilter(h, w)
        
        # YOLO 检测
        all_boxes = self._detect_boxes(img, conf)
        if verbose:
            print(f"检测到 {len(all_boxes)} 个候选框")
        
        if not all_boxes:
            return []
        
        # OCR + 符号检测 + 过滤
        results = []
        for i, (pts, det_conf) in enumerate(all_boxes):
            
            # 生长策略循环
            # 初始尝试：较小的 scale，避免截断边缘字符同时减少噪声
            # 如果截断，逐步扩大 (1.3 -> 1.8 -> 2.3)
            current_scale = 1.3
            ocr_text = ""
            ocr_conf = 0
            
            roi = None
            roi_for_ocr = None
            symbol_type = None
            symbol_bbox = None
            
            max_retries = 2
            for retry in range(max_retries + 1):
                # 1. 裁剪 ROI
                roi, rotation_info = self._get_smart_crop(img, pts, scale=current_scale)
                if roi is None or roi.size == 0:
                    break

                # 保存调试图片
                if ENABLE_DEBUG_SAVE_ROI and debug_dir and retry == 0:
                    debug_path = debug_dir / f"box_{i:03d}_scale_{current_scale:.1f}.png"
                    cv2.imwrite(str(debug_path), roi)

                # 放大图像
                roi_expanded = expand_image(roi, scale=2.5, min_height=64)
                
                # 2. 检测符号 (只在第一次或扩大后重新检测)
                # 如果之前已经检测到了，也许不需要重新检测？
                # 但扩大范围可能会把被截断的符号包进来，所以还是重新检测好
                symbol_type, symbol_bbox = self._detect_symbol(roi_expanded)
                
                # 3. 擦除符号 (根据开关)
                if ENABLE_SYMBOL_ERASE and symbol_type and symbol_bbox:
                    roi_for_ocr = self._erase_symbol_region(roi_expanded, symbol_bbox)
                else:
                    roi_for_ocr = roi_expanded
                
                # 4. OCR 识别
                new_text, new_conf = self._recognize_text(roi_for_ocr)

                # 4.5 修正 6/9 混淆（基于图像底部特征）
                if new_text and '6' in new_text:
                    new_text = correct_6_to_9(new_text, roi_for_ocr)

                # 5. 决策：是否接受新结果，是否继续生长
                should_grow = False
                
                # 如果是第一次尝试，直接接受
                if retry == 0:
                    ocr_text = new_text
                    ocr_conf = new_conf
                    
                    if not ocr_text:
                        should_grow = True # 空结果，必须生长
                    elif self._is_text_truncated(ocr_text):
                        should_grow = True # 疑似截断，尝试生长
                else:
                    # 如果不是第一次，只有当结果变得更好（更长）时才更新
                    if new_text and (not ocr_text or len(new_text) > len(ocr_text)):
                        if verbose:
                            print(f"  -> 生长成功 (scale={current_scale:.1f}): '{ocr_text}' -> '{new_text}'")
                        ocr_text = new_text
                        ocr_conf = new_conf
                        
                        # 如果新结果还是疑似截断，继续生长
                        if self._is_text_truncated(new_text):
                            should_grow = True
                
                if should_grow and retry < max_retries:
                    current_scale += 0.5 # 步进 0.5 (长得猛一点)
                    if verbose:
                        reason = "空结果" if not ocr_text else "疑似截断"
                        print(f"Box {i}: {reason} ('{ocr_text}'), 尝试扩大范围 (scale={current_scale:.1f})...")
                else:
                    break # 不需要生长或达到上限
            
            if not ocr_text:
                if verbose:
                    print(f"Box {i}: OCR 结果为空")
                continue
            
            # 4. 验证符号检测：如果 model_b 检测到 φ，但 OCR 文本不支持，则取消
            if symbol_type == "diameter":
                if not self._validate_diameter_detection(ocr_text):
                    symbol_type = None
                    symbol_bbox = None
            elif symbol_type == "thread":
                if not self._validate_thread_detection(ocr_text):
                    symbol_type = None
                    symbol_bbox = None
            
            # 5. 后备符号检测：如果 model_b 没检测到，但 OCR 文本包含 φ/R/M 符号
            if symbol_type is None:
                symbol_type = self._detect_symbol_from_text(ocr_text)
            
            if verbose:
                symbol_str = f" [{SYMBOL_PREFIX.get(symbol_type, '')}]" if symbol_type else ""
                print(f"Box {i}: '{ocr_text}'{symbol_str} (conf={ocr_conf:.2f})")
            
            # 解析
            dimension = parse_dimension_text(ocr_text, symbol_type)
            
            # 计算矩形边界
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            rect = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            
            # 过滤
            keep, reason = dim_filter.filter(dimension, rect)
            if not keep:
                if verbose:
                    print(f"  -> 过滤: {reason}")
                continue
            
            # 额外过滤：可疑小值（OCR 误识别检测）
            main_val = dimension.get("main_value")
            dim_type = dimension.get("dimension_type", "length")

            # 主值 <= 0.1 且是普通长度（不是直径/半径/螺纹），认为是误识别
            # 工程图中 0.1mm 独立尺寸几乎不存在，通常作为公差
            if main_val is not None and main_val <= 0.1 and dim_type == "length":
                if verbose:
                    print(f"  -> 过滤: TooSmallValue (val={main_val})")
                continue

            # 额外过滤：低置信度的可疑数值（针对中文误识别）
            # 当 OCR 置信度较低 且 主值为小数值时，很可能是中文被误识别
            # 例如："导光柱" -> "9.1"
            if main_val is not None and ocr_conf < 0.85 and dim_type == "length":
                # 主值在 1-20 范围内的小数值，没有公差，没有数量
                has_tol = dimension.get("tolerance", {}).get("type") != "none"
                has_qty = dimension.get("quantity")

                if 1 <= main_val <= 20 and not has_tol and not has_qty:
                    # 进一步检查：如果只是纯数字（如 "9.1"），且置信度 < 0.85
                    raw_text = dimension.get("raw_text", "")
                    text_check = raw_text.replace('.', '').replace(' ', '')
                    if text_check.isdigit():
                        if verbose:
                            print(f"  -> 过滤: LowConfSuspiciousValue (val={main_val}, conf={ocr_conf:.2f})")
                        continue
            
            # 保存结果
            dimension["rect"] = rect
            dimension["obb_points"] = pts.tolist()
            dimension["confidence"] = det_conf
            dimension["ocr_confidence"] = ocr_conf
            results.append(dimension)
            
            if verbose:
                print(f"  -> 保留: {dimension.get('main_value')}")
        
        # 去重（两层）
        results = self._remove_duplicates(results)
        results = self._remove_duplicates_by_value(results)
        if verbose:
            print(f"最终结果: {len(results)} 个尺寸")
        
        return results
    
    def _detect_boxes(self, img, conf):
        """检测所有尺寸框"""
        all_boxes = []
        pred = self.model_a.predict(img, save=False, conf=conf, verbose=False, imgsz=1280, device=self.model_a_device)[0]
        
        if pred.obb:
            for obb in pred.obb:
                if int(obb.cls[0]) != 0:
                    continue
                pts = obb.xyxyxyxy.cpu().numpy()[0]
                all_boxes.append((pts, float(obb.conf[0])))
        
        return all_boxes
    
    def _detect_symbol(self, roi):
        """检测符号类型（φ/R/M）并返回边界框 - 多角度检测
        
        竖排标注时符号可能是旋转的，因此需要多角度检测。
        为减少误检，非 0° 角度需要更高置信度。
        
        Returns:
            tuple: (symbol_type, bbox) 或 (None, None)
            bbox 格式: [x1, y1, x2, y2]（原始 ROI 坐标系）
        """
        # 角度及其对应的最低置信度阈值
        # 降低阈值以提高检出率（避免漏检 φ 符号）
        angle_configs = [
            (0, 0.4),    # 0° 用 0.4（从0.5降低，进一步提高检出）
            (90, 0.5),   # 旋转角度用稍高阈值
            (180, 0.5),
            (270, 0.5),
        ]
        
        best_result = (None, None)
        best_conf = 0
        
        h, w = roi.shape[:2]
        
        for angle, min_conf in angle_configs:
            try:
                # 旋转图像
                if angle == 0:
                    rotated = roi
                elif angle == 90:
                    rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(roi, cv2.ROTATE_180)
                else:
                    rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 检测（使用较低的基础阈值，后面再按角度过滤）
                pred = self.model_b.predict(rotated, save=False, conf=0.45, verbose=False, device=self.model_b_device)[0]
                
                for box in pred.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 检查是否满足该角度的最低置信度要求
                    # 对M符号(螺纹)提高要求，避免误检比例标注
                    required_conf = min_conf
                    if cls_id == 2:  # M符号（假设class_id=2是螺纹）
                        required_conf = max(min_conf, 0.6)  # M符号至少需要0.6置信度

                    if cls_id in SYMBOL_NAMES and conf >= required_conf and conf > best_conf:
                        # 获取边界框坐标（旋转后的坐标系）
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # 将边界框坐标转换回原始 ROI 坐标系
                        rh, rw = rotated.shape[:2]
                        if angle == 90:
                            # 90度顺时针：(x,y) -> (rh-1-y, x) 的逆变换
                            x1_orig = y1
                            y1_orig = rw - x2
                            x2_orig = y2
                            y2_orig = rw - x1
                        elif angle == 180:
                            x1_orig = w - x2
                            y1_orig = h - y2
                            x2_orig = w - x1
                            y2_orig = h - y1
                        elif angle == 270:
                            x1_orig = rh - y2
                            y1_orig = x1
                            x2_orig = rh - y1
                            y2_orig = x2
                        else:
                            x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
                        
                        # 确保坐标有效
                        x1_orig = max(0, min(x1_orig, w))
                        y1_orig = max(0, min(y1_orig, h))
                        x2_orig = max(0, min(x2_orig, w))
                        y2_orig = max(0, min(y2_orig, h))
                        
                        bbox = [min(x1_orig, x2_orig), min(y1_orig, y2_orig),
                                max(x1_orig, x2_orig), max(y1_orig, y2_orig)]
                        best_result = (SYMBOL_NAMES[cls_id], bbox)
                        best_conf = conf
            except:
                pass
        
        return best_result
    
    def _erase_symbol_region(self, roi, bbox, padding=2):
        """擦除符号区域（用白色填充）
        
        Args:
            roi: 图像区域
            bbox: 符号边界框 [x1, y1, x2, y2]
            padding: 额外扩展的像素
        
        Returns:
            擦除符号后的图像副本
        """
        if bbox is None:
            return roi
        
        roi_clean = roi.copy()
        x1, y1, x2, y2 = bbox
        
        # 扩展边界
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(roi.shape[1], x2 + padding)
        y2 = min(roi.shape[0], y2 + padding)
        
        # 用白色填充（假设工程图纸背景是白色）
        roi_clean[y1:y2, x1:x2] = 255
        
        return roi_clean
    
    def _detect_symbol_from_text(self, text):
        """从 OCR 文本中检测符号类型（后备方案）
        
        当 model_b 未检测到符号时，尝试从 OCR 文本中提取符号信息。
        
        Returns:
            str: "diameter", "radius", "thread" 或 None
        """
        if not text:
            return None
        
        # φ 符号的各种变体
        phi_chars = ['φ', 'Φ', 'ф', 'Ф', '⌀', '¢', '桅', '鈲', '须', '蠁']
        for c in phi_chars:
            if c in text:
                return "diameter"
        
        # R 符号（半径）- 必须是大写 R 开头后面跟数字
        if re.search(r'^R\s*\d', text) or re.search(r'[-xX×]\s*R\s*\d', text):
            return "radius"
        
        # M 符号（螺纹）- 必须是大写 M 开头后面跟数字
        if re.search(r'^M\s*\d', text) or re.search(r'[-xX×]\s*M\s*\d', text):
            return "thread"
        
        return None
    
    def _validate_diameter_detection(self, ocr_text):
        """验证 φ 符号检测是否可信
        
        当 model_b 检测到 φ 符号时，通过 OCR 文本验证是否真的存在 φ 符号。
        这可以减少误检（如数字 0、6、8 被误识别为 φ）。
        
        Args:
            ocr_text: OCR 识别的文本
            
        Returns:
            bool: True 表示检测可信，False 表示应该取消检测
        """
        if not ocr_text:
            return False
        
        text = ocr_text.strip()
        
        # 1. OCR 直接识别到 φ 相关字符 -> 可信
        phi_chars = ['φ', 'Φ', 'ф', 'Ф', '⌀', '¢', '桅', '鈲', '须', '蠁']
        for c in phi_chars:
            if c in text:
                return True
        
        # 2. 数量-尺寸格式：N-X.X 或 N- X.X -> 可信（φ 可能被擦除了）
        # 例如: "2- 2.5-0.1", "4 -4.5", "4- 1.6"
        if re.match(r'^\d+\s*[-xX×]\s*\d', text):
            return True
        
        # 2b. 特殊情况：数量 + 公差 + 0开头的数字（φ 被识别为 0）
        # 例如: "4 +0.03 -01.950" -> 实际是 4-φ1.95+0.03
        if re.search(r'[-\s]0\d+\.', text):
            return True
        
        # 3. 以 0 开头的数字（如 02.5）-> 可信（0 是 φ 的误识别）
        if re.match(r'^0\d', text):
            return True
        
        # 4. 纯数字或带公差的数字，没有数量前缀
        # 例如: "0.5", "11.86 +0.05 -0.03", "1.30 +0.2"
        # 这些情况下 model_b 可能是误检，但也可能是 OCR 漏掉了 φ 符号
        # 放宽条件：如果有公差（+0 -0.1 格式），且主值较小（<20），认为可信
        # 因为工程图中小直径（如 φ3.4）带公差的情况很常见
        if re.match(r'^[\d\.\s\+\-±]+$', text):
            # 检查是否有公差
            has_tolerance = '+' in text or '-' in text or '±' in text
            # 提取主值
            main_match = re.match(r'^(\d+\.?\d*)', text)
            if main_match:
                main_val = float(main_match.group(1))
                # 如果有公差且主值较小，认为可信（可能是 OCR 漏掉了 φ）
                if has_tolerance and main_val < 20:
                    return True
            return False
        
        # 其他情况默认可信
        return True
    
    def _validate_thread_detection(self, ocr_text):
        """验证 M (螺纹) 符号检测是否可信

        Args:
            ocr_text: OCR 识别的文本

        Returns:
            bool: True 表示检测可信，False 表示应该取消检测
        """
        if not ocr_text:
            return False

        text = ocr_text.strip().upper()

        # 0. 如果包含冒号，很可能是比例标注（如3.3:1），不是螺纹
        if ':' in text or '：' in text:
            return False

        # 1. OCR 明确包含 M -> 可信
        if 'M' in text:
            return True

        # 2. 包含"深"字，可能是螺纹孔 (如 "3深5") -> 稍微可信，但通常会有 M
        if '深' in text:
            return True

        # 3. 纯数字或带公差的数字，且没有 M -> 不可信
        # 螺纹标注通常必须包含 M (如 M6, M1.4)，纯数字 (如 1.9, 6) 绝大多数是长度或轴径
        # 尤其是带公差的数字 (如 6 -0.03, 1.90 +0.05)，这明显是配合公差，不是螺纹
        if re.match(r'^[\d\.\s\+\-±]+$', text):
            return False

        return True

    def _recognize_text(self, roi):
        """OCR 识别 - 优化版多角度测试

        优先级策略：
        1. 0° 和 180° 是主要角度（横排文字）
        2. 90° 和 270° 是备用角度（竖排文字）
        3. 只有当主要角度识别失败时，才使用备用角度
        """
        # 注意：roi 已经在外部放大过了，这里不再重复放大

        # 先测试主要角度（0°, 180°）
        primary_angles = [0, 180]
        best_text = ""
        best_conf = 0
        best_score = 0

        for angle in primary_angles:
            if angle == 0:
                rotated = roi
            else:
                rotated = cv2.rotate(roi, cv2.ROTATE_180)

            result = self.ocr.ocr(rotated, det=True, rec=True, cls=True)
            if result and result[0]:
                text, conf = merge_ocr_results(result)
                # 评分 = 置信度 × 数字长度 × 数字数量
                text_clean = ''.join(c for c in text if c.isdigit() or c == '.')
                digit_count = len([c for c in text if c.isdigit()])
                score = conf * len(text_clean) * (1 + digit_count * 0.1)

                if score > best_score:
                    best_score = score
                    best_conf = conf
                    best_text = text

        # 如果主要角度识别成功且置信度足够高，直接返回
        # 提高阈值：需要置信度 > 0.9 并且有多个数字字符（避免误识别单个数字）
        if best_text:
            digit_count = len([c for c in best_text if c.isdigit()])
            # 条件1: 置信度很高 (>0.95) 且有数字
            # 条件2: 置信度较高 (>0.9) 且有至少2个数字
            # 条件3: 文本中包含明确的符号标记（±, ×, φ, M等）
            has_symbols = any(s in best_text for s in ['±', '×', 'φ', 'Φ', 'M', 'R', '+', '-', '.'])

            if (best_conf > 0.95 and digit_count >= 1) or \
               (best_conf > 0.9 and digit_count >= 2) or \
               (best_conf > 0.85 and has_symbols and digit_count >= 1):
                return best_text, best_conf

        # 主要角度失败，尝试备用角度（90°, 270°）
        secondary_angles = [90, 270]

        for angle in secondary_angles:
            if angle == 90:
                rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

            result = self.ocr.ocr(rotated, det=True, rec=True, cls=True)
            if result and result[0]:
                text, conf = merge_ocr_results(result)
                text_clean = ''.join(c for c in text if c.isdigit() or c == '.')
                digit_count = len([c for c in text if c.isdigit()])
                score = conf * len(text_clean) * (1 + digit_count * 0.1)

                if score > best_score:
                    best_score = score
                    best_conf = conf
                    best_text = text

        return best_text, best_conf
    
    def _remove_duplicates(self, results, iou_threshold=0.3):
        """去重 - 基于 IoU 和包含关系"""
        if len(results) <= 1:
            return results
        
        # 按面积从小到大排序
        results = sorted(results, key=lambda r: r["rect"][2] * r["rect"][3])
        
        keep = []
        for r in results:
            r_rect = r["rect"]
            is_dup = False
            
            for kept in keep:
                k_rect = kept["rect"]
                
                # 检查 IoU
                iou = self._calc_iou(r_rect, k_rect)
                if iou > iou_threshold:
                    is_dup = True
                    break
                
                # 检查包含关系：如果中心点在已保留框内，也认为是重复
                r_cx = r_rect[0] + r_rect[2] / 2
                r_cy = r_rect[1] + r_rect[3] / 2
                k_cx = k_rect[0] + k_rect[2] / 2
                k_cy = k_rect[1] + k_rect[3] / 2
                
                # 新框中心在已保留框内
                if (k_rect[0] <= r_cx <= k_rect[0] + k_rect[2] and
                    k_rect[1] <= r_cy <= k_rect[1] + k_rect[3]):
                    # 如果主值相同或接近，认为重复
                    r_val = r.get("main_value", 0)
                    k_val = kept.get("main_value", 0)
                    if r_val and k_val:
                        if abs(r_val - k_val) < 0.5 or r_val == k_val:
                            is_dup = True
                            break
                
                # 已保留框中心在新框内
                if (r_rect[0] <= k_cx <= r_rect[0] + r_rect[2] and
                    r_rect[1] <= k_cy <= r_rect[1] + r_rect[3]):
                    r_val = r.get("main_value", 0)
                    k_val = kept.get("main_value", 0)
                    if r_val and k_val:
                        if abs(r_val - k_val) < 0.5 or r_val == k_val:
                            is_dup = True
                            break
            
            if not is_dup:
                keep.append(r)
        
        return keep
    
    def _remove_duplicates_by_value(self, results, distance_threshold=80):
        """第二层去重 - 基于主值和位置距离
        相同主值且位置接近的框，只保留置信度最高的
        """
        if len(results) <= 1:
            return results
        
        keep = []
        for r in results:
            r_rect = r["rect"]
            r_val = r.get("main_value", 0)
            r_cx = r_rect[0] + r_rect[2] / 2
            r_cy = r_rect[1] + r_rect[3] / 2
            r_conf = r.get("confidence", 0)
            
            is_dup = False
            dup_idx = -1
            
            for idx, kept in enumerate(keep):
                k_rect = kept["rect"]
                k_val = kept.get("main_value", 0)
                k_cx = k_rect[0] + k_rect[2] / 2
                k_cy = k_rect[1] + k_rect[3] / 2
                
                # 如果主值相同或非常接近
                if r_val and k_val and abs(r_val - k_val) < 0.1:
                    # 计算中心点距离
                    dist = ((r_cx - k_cx) ** 2 + (r_cy - k_cy) ** 2) ** 0.5
                    if dist < distance_threshold:
                        is_dup = True
                        dup_idx = idx
                        break
            
            if is_dup:
                # 如果新框置信度更高，替换已保留的
                k_conf = keep[dup_idx].get("confidence", 0)
                if r_conf > k_conf:
                    keep[dup_idx] = r
            else:
                keep.append(r)
        
        return keep
    
    def _calc_iou(self, rect1, rect2):
        """计算 IoU"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0


# ============== 输出格式化 ==============

def format_output(results):
    """格式化输出为 JSON 结构"""
    CATEGORY_NAMES = {
        "length": "长度",
        "diameter": "直径(φ)",
        "radius": "半径(R)",
        "thread": "螺纹(M)",
        "square": "正方形(□)",
        "rectangle": "矩形(×)"
    }
    
    output = []
    for i, r in enumerate(results):
        main_val = r.get("main_value", 0)
        secondary_val = r.get("secondary_value")  # 次值（矩形尺寸）
        depth = r.get("depth")                    # 深度值
        tol = r.get("tolerance", {})
        tol_type = tol.get("type", "none")
        upper = tol.get("upper") or 0
        lower = tol.get("lower") or 0
        dim_type = r.get("dimension_type", "length")
        rect = r.get("rect", [0, 0, 0, 0])
        quantity = r.get("quantity")
        
        prefix = SYMBOL_PREFIX.get(dim_type, "")
        
        # 构建显示文本
        if dim_type == "rectangle" and secondary_val:
            # 矩形尺寸：14.05×12.05
            text = f"{main_val}×{secondary_val}"
            if tol_type == "symmetric" and upper != 0:
                text += f"±{abs(upper)}"
            elif upper != 0 or lower != 0:
                text += f"/+{upper}/{lower}"
        elif tol_type == "symmetric" and upper != 0:
            # 对称公差用 ± 显示
            text = f"{prefix}{main_val}±{abs(upper)}"
        elif upper != 0 or lower != 0:
            # 非对称公差
            text = f"{prefix}{main_val}/+{upper}/{lower}"
        else:
            # 无公差
            text = f"{prefix}{main_val}"
        
        # 如果有深度，加上深度后缀
        if depth:
            text += f"深{depth}"
        
        # 如果有数量，加上数量前缀
        if quantity and quantity > 1:
            text = f"{quantity}-{text}"
        
        output.append({
            "id": i + 1,
            "text": text,
            "raw_text": r.get("raw_text", ""),
            "confidence": round(r.get("confidence", 0), 2),
            "ocr_confidence": round(r.get("ocr_confidence", 0), 2),
            "rect": {
                "x": rect[0], "y": rect[1],
                "width": rect[2], "height": rect[3]
            },
            "obb_points": r.get("obb_points", []),
            "category": CATEGORY_NAMES.get(dim_type, "长度"),
            "theoretical_value": main_val,
            "secondary_value": secondary_val,
            "depth": depth,                       # 新增：深度字段
            "upper_tolerance": upper,
            "lower_tolerance": lower,
            "quantity": quantity,
            "tolerance_type": tol_type
        })
    
    return output


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description='YOLO OCR 尺寸识别器 v2')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    parser.add_argument('--det-conf', '--conf', dest='conf', type=float, default=0.05, help='检测置信度阈值')
    parser.add_argument('--use-gpu', action='store_true', help='启用 GPU (默认使用 CPU)')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图片')
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: 图片不存在 {args.image}")
        sys.exit(1)

    # 默认使用 CPU (use_gpu=False)，只有加了 --use-gpu 才开启
    recognizer = DimensionRecognizer(use_gpu=args.use_gpu)
    results = recognizer.process_image(str(image_path), conf=args.conf)

    # 排序（Z型：先Y后X）
    results.sort(key=lambda r: (r.get("rect", [0, 0])[1], r.get("rect", [0, 0])[0]))

    print(f"\n识别到 {len(results)} 个尺寸")

    output_data = format_output(results)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {args.output}")

    if args.visualize:
        print("正在生成可视化结果...")
        # 默认保存为同名 png，位于 output 目录 (或者和 json 同目录)
        json_path = Path(args.output)
        vis_output = json_path.with_name(json_path.stem + "_visualized.png")
        visualize(str(image_path), str(json_path), str(vis_output))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
