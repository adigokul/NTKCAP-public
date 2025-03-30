from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal, Qt

class LabelClickable(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # only can be cjecked when cameras are opened and in multi person mode
        self._checkable = False 

    def setClickable(self, clickable=bool):
        self._checkable = clickable

    def isClickable(self):
        return self._checkable
    
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._checkable:
            self.clicked.emit()
        super().mousePressEvent(ev)