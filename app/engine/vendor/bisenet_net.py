"""
BiSeNet Face Parser - 完整独立实现
包含模型架构和解析器接口，不依赖外部仓库

19个类别 (CelebAMask-HQ):
0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict


# ============================================================================
# ResNet18 Backbone
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    """ResNet18 backbone for BiSeNet"""
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)   # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32


# ============================================================================
# BiSeNetV1 Components
# ============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


# ============================================================================
# BiSeNetV1 Model
# ============================================================================

class BiSeNetV1(nn.Module):
    """
    BiSeNetV1 面部解析模型
    
    Args:
        n_classes: 类别数量（默认19用于CelebAMask-HQ）
    """
    def __init__(self, n_classes=19):
        super(BiSeNetV1, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        
        return feat_out, feat_out16, feat_out32


# ============================================================================
# Face Parser Interface
# ============================================================================

class BiSeNetFaceParser:
    """
    BiSeNet 面部解析器
    
    19个类别:
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
    12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    """
    
    CLASS_NAMES = [
        'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
        'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
        'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
    ]
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        初始化 BiSeNet 面部解析器
        
        Args:
            model_path: 模型文件路径 (.pth)
            device: 计算设备 ('cuda' 或 'cpu')
        """
        if model_path is None:
            # 尝试多个可能的模型文件名
            # 权重平铺在 <仓库根>/weights/bisenet_model_final.pth（老位置仍作兜底）
            _weights_root = Path(__file__).resolve().parents[3] / 'weights'
            possible_paths = [
                _weights_root / 'bisenet_model_final.pth',
                Path(__file__).parent / 'model_final.pth',
                Path(__file__).parent / '79999_iter.pth',
                Path(__file__).parent / 'pytorch_model.bin',
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    print(f"Found model: {path.name}")
                    break
            else:
                model_path = str(possible_paths[0])
                print(f"Warning: No model found, using default path: {model_path}")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"BiSeNet using device: {self.device}")
        print(f"Loading model from: {model_path}")
        
        # 创建 BiSeNetV1 模型
        n_classes = 19
        self.model = BiSeNetV1(n_classes=n_classes)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 移除可能的 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.n_classes = n_classes
        self.input_size = 512
        print(f"✓ BiSeNet loaded successfully")

    def parse(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对输入图像进行面部解析
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            包含各个类别掩码的字典（与输入图像尺寸相同）
        """
        # 保存原始尺寸
        original_h, original_w = image.shape[:2]
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)[0]
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # 调整预测结果到原始尺寸
        if pred.shape != (original_h, original_w):
            pred_resized = cv2.resize(pred.astype(np.uint8), (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            pred_resized = pred
        
        # 后处理：生成各类别掩码
        masks = {}
        for idx, class_name in enumerate(self.CLASS_NAMES):
            mask = (pred_resized == idx).astype(np.uint8) * 255
            masks[class_name] = mask
        
        return masks

    def get_face_mask(self, image: np.ndarray) -> np.ndarray:
        """
        获取完整的面部区域掩码（包含皮肤、五官、头发等）
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            面部区域二值掩码 (H, W)
        """
        masks = self.parse(image)
        
        # 合并所有面部相关类别
        face_classes = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 
                       'nose', 'mouth', 'u_lip', 'l_lip', 'l_ear', 'r_ear', 'hair']
        
        face_mask = np.zeros_like(masks['background'])
        for class_name in face_classes:
            if class_name in masks:
                face_mask = np.maximum(face_mask, masks[class_name])
        
        return face_mask

    def get_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        获取头发区域掩码
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            头发区域二值掩码 (H, W)
        """
        masks = self.parse(image)
        return masks.get('hair', np.zeros_like(masks['background']))

    def get_face_and_hair_masks(self, image: np.ndarray) -> tuple:
        """
        获取面部和头发的分离掩码
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            (face_mask, hair_mask) 元组
            - face_mask: 面部区域二值掩码 (H, W)，值为 0 或 255
            - hair_mask: 头发区域二值掩码 (H, W)，值为 0 或 255
        """
        masks = self.parse(image)
        h, w = masks['background'].shape
        
        # 面部区域
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_classes = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 
                       'nose', 'mouth', 'u_lip', 'l_lip', 'l_ear', 'r_ear']
        for class_name in face_classes:
            if class_name in masks:
                face_mask = np.maximum(face_mask, masks[class_name])
        
        # 头发区域
        hair_mask = masks.get('hair', np.zeros((h, w), dtype=np.uint8))
        
        return face_mask, hair_mask

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理
        
        Args:
            image: BGR 图像 (H, W, 3)
            
        Returns:
            预处理后的张量 (1, 3, H, W)
        """
        # 调整大小到 512x512
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # BGR to RGB
        img_rgb = img_resized[:, :, ::-1]
        
        # 归一化到 [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # 标准化 (ImageNet mean and std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # HWC to CHW
        img_transposed = img_normalized.transpose(2, 0, 1)
        
        # 添加 batch 维度
        img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)
        
        return img_tensor.to(self.device)


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("BiSeNetV1 Standalone Test")
    print("=" * 80)
    
    # 测试模型架构
    net = BiSeNetV1(19)
    net.eval()
    in_ten = torch.randn(1, 3, 512, 512)
    out, out16, out32 = net(in_ten)
    print(f"\n✓ Model architecture test passed!")
    print(f"  Output shape: {out.shape}")
    print(f"  Output16 shape: {out16.shape}")
    print(f"  Output32 shape: {out32.shape}")
    
    # 测试解析器（如果有模型文件）
    model_path = Path(__file__).parent / 'model_final.pth'
    if model_path.exists():
        print(f"\n✓ Testing face parser...")
        parser = BiSeNetFaceParser(str(model_path))
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (446, 446, 3), dtype=np.uint8)
        masks = parser.parse(test_image)
        print(f"  Generated {len(masks)} masks")
        print(f"✓ Face parser test passed!")
    else:
        print(f"\n⚠ No model file found, skipping parser test")
