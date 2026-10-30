from PyQt5.QtWidgets import QVBoxLayout, QWidget

class SiDenseVContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
    def addWidget(self, widget, alignment=0):
        self.layout.addWidget(widget, alignment=alignment)
