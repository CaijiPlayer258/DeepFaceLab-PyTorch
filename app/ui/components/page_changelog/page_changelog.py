"""更新日志页面"""
from pathlib import Path
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt
from siui.components.page import SiPage
from siui.components import SiTitledWidgetGroup
from siui.components.container import SiTriSectionPanelCard
from siui.core import SiGlobal, SiColor


class ChangelogPage(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPadding(64)
        self.setScrollMaximumWidth(1000)
        self.setScrollAlignment(Qt.AlignLeft)
        self.setTitle("更新日志")

        _log_path = Path(__file__).parent.parent.parent.parent / "updata.txt"
        _lines = []
        if _log_path.exists():
            with open(str(_log_path), 'r', encoding='utf-8') as f:
                for l in f:
                    l = l.strip()
                    if l:
                        _lines.append(l)

        _text = '\n'.join(_lines)

        _group = SiTitledWidgetGroup(self)
        with _group as g:
            g.addTitle(f"更新日志（共 {len(_lines)} 条）")
            _card = SiTriSectionPanelCard(g)
            _card.setTitle("版本历史")
            _edit = QTextEdit()
            _edit.setReadOnly(True)
            _edit.setPlainText(_text if _text else "暂无更新记录")
            _edit.setMinimumHeight(400)
            _edit.setStyleSheet(
                "QTextEdit { background: transparent; border: none;"
                "  color: #FFFFFF; font-size: 12px;"
                "  font-family: 'Inter', 'Consolas', monospace;"
                "}"
                "QScrollBar:vertical { background: transparent; width: 6px; border: none; }"
                "QScrollBar::handle:vertical { background: #3a3a52; border-radius: 3px; min-height: 30px; }"
                "QScrollBar::handle:vertical:hover { background: #5a5a72; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
                "QScrollBar:horizontal { height: 0; }"
            )
            _card.body().addWidget(_edit)
            _card.adjustSize()
            g.addWidget(_card)

        self.setAttachment(_group)
