# Training Config Pages - 模块化结构说明

## 目录结构

```
ui/components/page_trainer/
├── page_trainer.py                    # 主页面（TrainerPage）
├── components/                        # 子页面组件目录
│   ├── __init__.py                    # 导出类
│   ├── training_config_page.py        # 训练配置子页面
│   └── new_model_config_page.py       # 新建模型配置子页面
└── README.md                          # 本文档
```

## 文件说明

### 1. `page_trainer.py` - 主页面
- **类**: `TrainerPage`
- **功能**: Trainer的主页面，包含水平导航栏和模型列表
- **导入**: 从 `components` 目录导入两个子页面类
- **方法**:
  - `open_training_config(model_name)`: 打开训练配置子页面
  - `open_new_model_config(model_name)`: 打开新建模型配置子页面

### 2. `components/training_config_page.py` - 训练配置子页面
- **类**: `TrainingConfigChildPage`
- **触发时机**: 点击现有模型的"训练"按钮时弹出
- **特点**:
  - 包含完整的训练配置（数据集、训练参数、优化器等）
  - archi 字段为**不可编辑的文本框**，显示格式如 "df-ud"
  - 因为模型已创建，架构不能修改
  - 底部按钮："开始训练" 和 "取消"

### 3. `components/new_model_config_page.py` - 新建模型配置子页面
- **类**: `NewModelConfigChildPage`
- **触发时机**: 点击“新建模型”按钮时弹出
- **特点**:
  - 仅包含模型架构相关参数
  - archi 和 subarchi 为**可选择的胶囊下拉框**
  - archi 选项: DF, LIAE, AMP
  - subarchi 选项: -u, -ud, -ut, -udt, -d, -dt, -t
  - 选择不同选项时，卡片副标题会动态更新
  - 底部按钮：“创建模型” 和 “取消”
- **信号**:
  - `model_created`: 当用户点击“创建模型”按钮时发出，传递配置数据字典

## 使用示例

### 在主页面中打开子页面

```python
from ui.components.page_trainer.page_trainer import TrainerPage

class MyWindow:
    def on_train_clicked(self, model_name):
        """点击训练按钮"""
        self.open_training_config(model_name)
    
    def on_new_model_clicked(self):
        """点击新建模型按钮"""
        new_model_name = f"model_{self.model_counter:03d}"
        self.model_counter += 1
        self.open_new_model_config(new_model_name)
```

### 直接导入子页面类

```python
from ui.components.page_trainer.components import (
    TrainingConfigChildPage,
    NewModelConfigChildPage
)

# 创建训练配置页面
training_page = TrainingConfigChildPage(model_name="my_model")

# 创建新建模型配置页面
new_model_page = NewModelConfigChildPage(model_name="new_model")
```

## 设计原则

### 1. 模块化
- 每个子页面独立为一个文件
- 便于维护和扩展
- 参考 `page_dialog` 的组织方式

### 2. 职责分离
- **TrainingConfigChildPage**: 用于训练已有模型，架构不可修改
- **NewModelConfigChildPage**: 用于创建新模型，需要选择架构

### 3. 用户体验
- 训练页面：archi 显示为不可编辑状态，防止误操作
- 新建页面：archi 和 subarchi 可选择，并显示详细描述

### 4. 信号机制
- `NewModelConfigChildPage` 通过 `model_created` 信号通知父页面
- `TrainerPage` 监听信号并执行以下操作：
  1. 从配置数据中提取模型参数
  2. 创建新的模型选项卡（**包含淡入动画**）
  3. 延迟100ms后更新副标题格式：`archi[df-ud]--resolution[256]--ae[256]--e[64]--d[128]--d_mask[32]--iter[1500000]`
  4. 将新卡片插入到“新建模型”按钮之前

### 5. 动画效果
- **淡入动画**: 新创建的模型卡片会从透明渐变到不透明
  - 使用 `QGraphicsOpacityEffect` 控制透明度
  - 使用 `QVariantAnimation` 实现平滑过渡
  - 动画时长: 300ms
  - 缓动曲线: OutCubic（先快后慢）
  - 透明度范围: 0.0 → 1.0
- **延迟更新**: 副标题在动画开始100ms后更新，确保视觉效果流畅

### 6. 刷新机制
插入新卡片后，调用已有的 `_animate_models_card_height()` 方法来处理布局刷新：

```python
# 触发卡片高度动画（会自动处理布局刷新）
self._animate_models_card_height()
```

**工作原理**：
1. 调用 `layout().invalidate()` 和 `layout().activate()` 强制重新计算布局
2. 使用 `SiExpAnimation` 平滑动画调整卡片高度
3. 在动画每一帧都更新卡片高度，同时保持宽度不变
4. 自动处理所有必要的几何更新和重绘

**优势**：
- 复用项目中已有的成熟代码
- 包含平滑的高度动画效果
- 自动处理布局计算和刷新
- 滚动时卡片位置正确

## 后续优化建议

1. **保存用户偏好**
   - 记住用户上次选择的架构
   - 为不同架构预设不同的默认参数

2. **添加架构对比功能**
   - 在新建模型页面添加“查看架构对比”按钮
   - 显示不同架构的特点和适用场景

3. **完善 TrainingConfigChildPage**
   - 添加完整的训练参数配置卡片（已包含）
   - 确保所有参数都能正确保存和加载

## 测试

运行测试脚本验证模块化结构：

```bash
python test_modular_training_pages.py
```

预期输出：
```
============================================================
测试模块化结构
============================================================

1. 创建 TrainingConfigChildPage...
✓ TrainingConfigChildPage 创建成功
  - archi 输入框文本: df-ud
  - archi 输入框是否可编辑: False

2. 创建 NewModelConfigChildPage...
✓ NewModelConfigChildPage 创建成功
  - archi 下拉框选项数: 3
  - 子分支下拉框选项数: 7

============================================================
✓ 所有测试通过！
============================================================
```
