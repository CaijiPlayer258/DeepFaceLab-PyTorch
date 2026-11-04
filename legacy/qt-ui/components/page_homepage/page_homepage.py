
from PyQt5.QtCore import Qt
from siui.components import (
    SiCircularProgressBar,
    SiDenseVContainer,
    SiLineEditWithDeletionButton,
    SiLineEditWithItemName,
    SiOptionCardLinear,
    SiTitledWidgetGroup,
    SiFlowContainer,
    SiWidget,
)
from siui.components import SiPixLabel
from siui.components.option_card import SiOptionCardLinear, SiOptionCardPlane
from siui.components.page import SiPage
from siui.components.slider import SiSliderH
from siui.components.titled_widget_group import SiTitledWidgetGroup
from siui.components.widgets import (
    SiDenseHContainer,
    SiDenseVContainer,
    SiLabel,
    SiLineEdit,
    SiLongPressButton,
    SiPushButton,
    SiSimpleButton,
    SiSwitch,
)
from siui.core import GlobalFont, Si, SiColor, SiGlobal, SiQuickEffect, GlobalFontSize
from siui.gui import SiFont
from .components.themed_option_card import ThemedOptionCardPlane
class ExampleHomepage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import os
        
        # 获取项目根目录的绝对路径（page_homepage.py 在 ui/components/page_homepage/ 目录下）
        from pathlib import Path as _Path
        self.project_root = str(_Path(__file__).resolve().parent.parent.parent.parent)
        print("Current working directory:", os.getcwd())
        print("Project root:", self.project_root)
        
        # 滚动区域
        self.scroll_container = SiTitledWidgetGroup(self)
        # 整个顶部
        self.head_area = SiLabel(self)
        self.head_area.setFixedHeight(450)
        # 创建背景底图和渐变
        self.background_image = SiPixLabel(self.head_area)
        self.background_image.setFixedSize(1366, 300)
        self.background_image.setBorderRadius(6)
        
        # 使用绝对路径加载背景图片
        bg_image_path = os.path.join(self.project_root, 'ui', 'img', 'homepage_background.png')
        self.background_image.load(bg_image_path)
        self.background_fading_transition = SiLabel(self.head_area)
        self.background_fading_transition.setGeometry(0, 100, 0, 200)
        self.background_fading_transition.setStyleSheet(
            """
            background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0, stop:0 {}, stop:1 {})
            """.format(SiGlobal.siui.colors["INTERFACE_BG_B"],
                       SiColor.trans(SiGlobal.siui.colors["INTERFACE_BG_B"], 0))
        )
        # 创建大标题和副标题
        self.title = SiLabel(self.head_area)
        self.title.setGeometry(64, 0, 500, 128)
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.title.setText("深变:DFL-PyTorch")
        self.title.setStyleSheet("color: {}".format(SiGlobal.siui.colors["TEXT_A"]))
        self.title.setFont(SiFont.tokenized(GlobalFont.XL_MEDIUM))
        self.subtitle = SiLabel(self.head_area)
        self.subtitle.setGeometry(64, 72, 500, 48)
        self.subtitle.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.subtitle.setText("A modern and powerful DeepFake workflow")
        self.subtitle.setStyleSheet("color: {}".format(SiColor.trans(SiGlobal.siui.colors["TEXT_A"], 0.9)))
        self.subtitle.setFont(SiFont.tokenized(GlobalFont.S_MEDIUM))
        # 创建一个水平容器
        self.container_for_cards = SiDenseHContainer(self.head_area)
        self.container_for_cards.move(0, 130)
        self.container_for_cards.setFixedHeight(310)
        self.container_for_cards.setAlignment(Qt.AlignCenter)
        self.container_for_cards.setSpacing(32)
        # 添加卡片
        self.option_card_project = ThemedOptionCardPlane(self)
        self.option_card_project.setTitle("GitHub page")
        self.option_card_project.setFixedSize(218, 270)
        self.option_card_project.setThemeColor("#855198")
        self.option_card_project.setDescription(
            "本项目GitHub页面，欢迎前往查看源码、提交Issue")
        self.option_card_project.setURL("https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch")
        self.option_card_example = ThemedOptionCardPlane(self)
        self.option_card_example.setTitle("菜级玩家")
        self.option_card_example.setFixedSize(218, 270)
        self.option_card_example.setThemeColor("#7573aa")
        self.option_card_example.setDescription("菜级玩家是深变项目的作者，只在B站发布视频和软件，欢迎前往空间观看、收藏相关教学视频")  # noqa: E501
        self.option_card_example.setURL("https://space.bilibili.com/500398541")
        # 添加到水平容器
        self.container_for_cards.addPlaceholder(64 - 32)
        self.container_for_cards.addWidget(self.option_card_project)
        self.container_for_cards.addWidget(self.option_card_example)
        # 添加到滚动区域容器
        self.scroll_container.addWidget(self.head_area)
        
        # 创建流式容器（直接添加到页面，而不是滚动容器）
        self.flow_container = SiFlowContainer(self)
        self.flow_container.setGeometry(40, 450, self.width(), 2000)  # x=60, y=head_area 高度，width=窗口宽度 -120
        self.flow_container.setLineHeight(80)
        self.flow_container.setSiliconWidgetFlag(Si.EnableAnimationSignals)
        # 创建8个独立的选项卡实例
        self.option_card_1 = SiOptionCardLinear(self)
        self.option_card_1.setTitle("菜级玩家", "项目负责")
        self.option_card_1.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_1.setFixedWidth(350)
        self.option_card_1.adjustSize()
        
        self.option_card_2 = SiOptionCardLinear(self)
        self.option_card_2.setTitle("iperov", "原DeepFaceLab项目作者")
        self.option_card_2.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_2.setFixedWidth(350)
        self.option_card_2.adjustSize()
        
        self.option_card_3 = SiOptionCardLinear(self)
        self.option_card_3.setTitle("霏泠Ice", "组件库PyQt-SiliconUI的作者")
        self.option_card_3.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_3.setFixedWidth(350)
        self.option_card_3.adjustSize()
        
        self.option_card_4 = SiOptionCardLinear(self)
        self.option_card_4.setTitle("王富贵", "一只可爱的猫猫o(=•ェ•=)m")
        self.option_card_4.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_4.setFixedWidth(350)
        self.option_card_4.adjustSize()
        
        self.option_card_5 = SiOptionCardLinear(self)
        self.option_card_5.setTitle("my-bug", "本项目基于my-bug的版本完善")
        self.option_card_5.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_5.setFixedWidth(350)
        self.option_card_5.adjustSize()

        self.option_card_6 = SiOptionCardLinear(self)
        self.option_card_6.setTitle("鸦", "提供各种变体模型")
        self.option_card_6.load(SiGlobal.siui.iconpack.get("ic_fluent_card_ui_regular"))
        self.option_card_6.setFixedWidth(350)
        self.option_card_6.adjustSize()
        
        # 将 8 个独立选项卡添加到流式容器
        self.flow_container.addWidget(self.option_card_1)
        self.flow_container.addWidget(self.option_card_2)
        self.flow_container.addWidget(self.option_card_3)
        self.flow_container.addWidget(self.option_card_4)
        self.flow_container.addWidget(self.option_card_5)
        self.flow_container.addWidget(self.option_card_6)

        # 不再将流式容器添加到滚动区域，它独立存在
        # SiQuickEffect.applyDropShadowOn(self.container_for_cards, color=(0, 0, 0, 80), blur_radius=48)
        # 开始搭建界面
        self.version = SiLabel(self)
        self.version.setGeometry(40, self.height() - 60, 500, 48)
        self.version.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.version.setText("""version:v4.0.0""")
        self.version.setStyleSheet("color: {}".format(SiColor.trans(SiGlobal.siui.colors["TEXT_A"], 0.9)))
        self.version.setFont(SiFont.tokenized(GlobalFont.S_MEDIUM))
        # 添加到页面
        self.setAttachment(self.scroll_container)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        
        self.scroll_container.setFixedWidth(w)
        self.background_image.setFixedWidth(w)
        # self.titled_widget_group.setFixedWidth(min(w - 128, 900))  # Removed as titled_widget_group is not defined
        self.background_fading_transition.setFixedWidth(w)
        
        # 调整流式容器的宽度和位置
        self.flow_container.setGeometry(40, 450,w, self.flow_container.height())
        self.flow_container.arrangeWidgets(ani=True)
        self.version.setGeometry(40, self.height() - 60, 500, 48)
