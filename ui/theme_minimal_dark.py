# -*- coding: utf-8 -*-
"""极简暗色主题（ui-dark 分支）

设计语言：taste-skill/minimalist 的暗色适配
- 画布 #101013 / 表面 #141418 / 悬浮 #191A1D（无紫调、无渐变）
- 边框/分隔逻辑收敛到界面层级色差，强调色唯一：去饱和青灰 #6F8F90
- 文字永不用纯黑/纯白：主 #D9DADB → 弱 #68696E
- 状态色降饱和：成功 #5F8F6A / 警告 #A98B5F / 错误 #A65D5D

用法：在 ui/start.py 里 `import siui` 之后、构建窗口之前调用 apply()。
只覆写 SiliconUI 全局色组的 token，不改任何组件代码；删掉本文件的挂载即回原样。
"""
from siui.core import SiColor, SiGlobal

_TOKENS = {
    # 主题强调色：紫 → 去饱和青灰
    SiColor.THEME:                   '#6F8F90',
    SiColor.THEME_TRANSITION_A:      '#557071',
    SiColor.THEME_TRANSITION_B:      '#638485',
    SiColor.SVG_THEME:               '#6F8F90',
    SiColor.TEXT_THEME:              '#6F8F90',

    # SVG / 提示
    SiColor.SVG_NORMAL:              '#C0C1C4',
    SiColor.TOOLTIP_BG:              '#E01D1E21',
    SiColor.LAYER_DIM:               '#90000000',

    # 界面层级：紫灰调 → 中性近黑
    SiColor.INTERFACE_BG_A:          '#101013',
    SiColor.INTERFACE_BG_B:          '#141418',
    SiColor.INTERFACE_BG_C:          '#191A1D',
    SiColor.INTERFACE_BG_D:          '#1E1F24',
    SiColor.INTERFACE_BG_E:          '#242529',

    # 文字：离白 → 弱灰
    SiColor.TEXT_A:                  '#D9DADB',
    SiColor.TEXT_B:                  '#C0C1C4',
    SiColor.TEXT_C:                  '#939498',
    SiColor.TEXT_D:                  '#7C7D82',
    SiColor.TEXT_E:                  '#68696E',

    # 标题
    SiColor.TITLE_INDICATOR:         '#6F8F90',
    SiColor.TITLE_HIGHLIGHT:         '#1E2829',

    # 侧边消息
    SiColor.SIDE_MSG_FLASH:          '#90FFFFFF',
    SiColor.SIDE_MSG_THEME_NORMAL:   '#242529',
    SiColor.SIDE_MSG_THEME_SUCCESS:  '#5F8F6A',
    SiColor.SIDE_MSG_THEME_INFO:     '#6F8F90',
    SiColor.SIDE_MSG_THEME_WARNING:  '#A98B5F',
    SiColor.SIDE_MSG_THEME_ERROR:    '#A65D5D',

    SiColor.MENU_BG:                 '#191A1D',

    # 按钮
    SiColor.BUTTON_PANEL:            '#1F2023',
    SiColor.BUTTON_SHADOW:           '#0B0B0C',
    SiColor.BUTTON_THEMED_BG_A:      '#557071',
    SiColor.BUTTON_THEMED_BG_B:      '#638485',
    SiColor.BUTTON_THEMED_SHADOW_A:  '#141D1C',
    SiColor.BUTTON_THEMED_SHADOW_B:  '#17231F',
    SiColor.BUTTON_ON:               '#141D1C',
    SiColor.BUTTON_OFF:              '#242529',
    SiColor.BUTTON_TEXT_BUTTON_IDLE: '#6F8F90',
    SiColor.BUTTON_TEXT_BUTTON_FLASH:'#6F8F90',
    SiColor.BUTTON_TEXT_BUTTON_HOVER:'#A8C4C5',

    # 单选 / 复选
    SiColor.RADIO_BUTTON_UNCHECKED:  '#141418',
    SiColor.RADIO_BUTTON_CHECKED:    '#6F8F90',
    SiColor.CHECKBOX_SVG:            '#101013',
    SiColor.CHECKBOX_UNCHECKED:      '#939498',
    SiColor.CHECKBOX_CHECKED:        '#6F8F90',

    # 长按按钮：降饱和暖灰琥珀
    SiColor.BUTTON_LONG_PRESS_PANEL:   '#4A4340',
    SiColor.BUTTON_LONG_PRESS_SHADOW:  '#332E2B',
    SiColor.BUTTON_LONG_PRESS_PROGRESS:'#C9A66B',

    # 开关
    SiColor.SWITCH_DEACTIVATE:       '#3A3B3F',
    SiColor.SWITCH_ACTIVATE:         '#101013',

    # 滚动条 / 进度条
    SiColor.SCROLL_BAR:                  '#30FFFFFF',
    SiColor.PROGRESS_BAR_TRACK:          '#141418',
    SiColor.PROGRESS_BAR_PROCESSING:     '#6F8F90',
    SiColor.PROGRESS_BAR_COMPLETING:     '#5F8F6A',
    SiColor.PROGRESS_BAR_PAUSED:         '#68696E',
    SiColor.PROGRESS_BAR_FLASHES:        '#FFFFFF',
}


def apply():
    """把极简暗色 tokens 覆写到 SiliconUI 全局色组。"""
    applied = 0
    for token, code in _TOKENS.items():
        try:
            SiGlobal.siui.colors.assign(token, code)
            applied += 1
        except Exception as e:
            print(f'[theme] {token.name}: {e}')
    print(f'[theme] 极简暗色主题已挂载（{applied}/{len(_TOKENS)} tokens）')