"""City3D load-tab layout repair. Reuses existing widgets and signal connections."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout,
    QLayout, QSplitter, QScrollArea, QSizePolicy, QLabel, QPushButton, QApplication)


def _vertical_rows(layout):
    # A single narrow column avoids minimum-width overflow from long toolbars.
    if isinstance(layout, QHBoxLayout):
        layout.setDirection(QBoxLayout.TopToBottom)
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.layout():
            _vertical_rows(item.layout())


def repair(tab):
    if getattr(tab, '_layout_repaired_20261016', False):
        return
    root = tab.layout()
    original = next((root.itemAt(i).widget() for i in range(root.count())
                     if isinstance(root.itemAt(i).widget(), QSplitter)
                     and root.itemAt(i).widget().indexOf(tab.viewer) >= 0), None)
    if original is None or original.count() != 2:
        raise RuntimeError('Вкладка загрузки имеет неизвестную компоновку; нужен анализ её версии.')
    top = original.widget(0)
    items = []
    while root.count():
        items.append(root.takeAt(0))
    content = QWidget()
    content.setObjectName('city3d_load_controls')
    controls = QVBoxLayout(content)
    controls.setSizeConstraint(QLayout.SetMinAndMaxSize)
    for item in items:
        if item.widget() is original:
            continue
        if item.layout():
            controls.addLayout(item.layout())
        elif item.widget():
            controls.addWidget(item.widget())
        else:
            controls.addItem(item)
    controls.addWidget(top)
    _vertical_rows(controls)
    # Root rows from earlier patches now scroll; the existing viewer stays live.
    scroll = QScrollArea()
    scroll.setObjectName('city3d_load_scroll')
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    scroll.setMinimumSize(200, 100)
    tab.table.setMinimumHeight(220)
    tab.table.setMaximumHeight(340)
    tab.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tab.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tab.table.setColumnWidth(0, 280)
    tab.table.setColumnWidth(5, 140)
    tab.table.setToolTip('Флажок «В работу» включает файл в район. Выделение строки показывает её CRS. '
                         'После изменения флажков нажмите «Применить отмеченные LAS».')
    tab.file_path.setToolTip('Путь выделенной строки или применённого рабочего LAS; состав района задают флажки.')
    viewer = tab.viewer
    viewer.setMinimumSize(0, 0)
    viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    split = QSplitter(Qt.Horizontal)
    split.setObjectName('city3d_load_splitter')
    split.addWidget(scroll)
    split.addWidget(viewer)
    split.setChildrenCollapsible(False)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setSizes([520, 600])
    root.addWidget(split, 1)
    # Move, don't recreate: load and cancel retain their original connected slots.
    root.addWidget(tab.btn_load)
    tab.btn_load.setText('Применить отмеченные LAS → все вкладки')
    tab.btn_load.setMinimumWidth(0)
    tab.btn_load.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    footer = QHBoxLayout()
    tab.status.setMinimumWidth(0)
    tab.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    tab.status.setWordWrap(True)
    footer.addWidget(tab.status, 1)
    footer.addWidget(tab.progress)
    cancel = next((b for b in content.findChildren(QPushButton) if b.text() == 'Отмена'), None)
    if cancel:
        footer.addWidget(cancel)
    root.addLayout(footer)
    original.setParent(None)
    original.deleteLater()
    tab._layout_repaired_20261016 = True
    tab._load_scroll = scroll
    tab._load_splitter = split


def install(ns):
    base = ns['LoadDataTab']
    class LoadTab(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            repair(self)
    ns['LoadDataTab'] = LoadTab
    base_window = ns['MainWindow']
    class Window(base_window):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setMinimumSize(900, 600)
            self.tabs.setUsesScrollButtons(True)
            self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            screen = self.screen() or QApplication.primaryScreen()
            if screen:
                available = screen.availableGeometry()
                self.setMinimumSize(min(900, available.width()), min(600, available.height()))
                self.resize(min(1500, available.width()), min(950, available.height()))
    ns['MainWindow'] = Window
