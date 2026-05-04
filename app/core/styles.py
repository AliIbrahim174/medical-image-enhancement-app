"""Shared palette and Qt style sheets for the MedVision Workbench UI."""

BG0 = "#0c0d11"
BG1 = "#11131a"
BG2 = "#161923"
BG3 = "#1c2030"
BG4 = "#222840"
BORDER = "#252d42"
BORDER2 = "#2e3850"
ACCENT = "#4f9cf9"
ACCENT2 = "#3b7de8"
ACCENT_DIM = "#1a3a6e"
CYAN = "#2dd4bf"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"
TEXT0 = "#e2e8f4"
TEXT1 = "#9aaec8"
TEXT2 = "#5a6e88"
TEXT3 = "#3a4d64"

DARK_STYLE = f"""
QMainWindow, QWidget {{
    background: {BG0};
    color: {TEXT0};
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 12px;
}}
QMenuBar {{
    background: #090a0e;
    border-bottom: 1px solid {BORDER};
    spacing: 2px;
}}
QMenuBar::item {{
    background: transparent;
    color: {TEXT1};
    padding: 5px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background: {BG3};
    color: {TEXT0};
}}
QMenu {{
    background: {BG1};
    color: {TEXT1};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background: {BG3};
    color: {TEXT0};
}}
QToolTip {{
    background: #1e2535;
    color: {TEXT0};
    border: 1px solid {BORDER2};
    padding: 4px 6px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BG4};
    border-radius: 3px;
    min-height: 18px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BG4};
    border-radius: 3px;
    min-width: 18px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG0};
}}
QTabBar::tab {{
    background: {BG1};
    color: {TEXT2};
    padding: 6px 14px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
    font-size: 10.5px;
}}
QTabBar::tab:hover {{
    background: {BG2};
    color: {TEXT1};
}}
QTabBar::tab:selected {{
    background: {BG0};
    color: {ACCENT};
    border-color: {BORDER};
}}
QStatusBar {{
    background: #080a0e;
    border-top: 1px solid {BORDER};
    color: {TEXT2};
    font-size: 10px;
}}
QProgressBar {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {ACCENT2};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    background: {BG4};
    height: 3px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 10px;
    height: 10px;
    margin: -4px 0;
    border-radius: 5px;
}}
QLabel {{
    color: {TEXT1};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG3};
    color: {TEXT0};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {BG2};
    color: {TEXT0};
    selection-background-color: {ACCENT_DIM};
}}
"""

SIDEBAR_STYLE = f"""
QScrollArea {{
    background: {BG1};
    border: none;
}}
QToolButton, QPushButton {{
    background: {BG3};
    color: {TEXT1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
}}
QToolButton:hover, QPushButton:hover {{
    background: {BG4};
    color: {TEXT0};
    border-color: {BORDER2};
}}
QToolButton:checked {{
    background: {ACCENT_DIM};
    color: {ACCENT};
    border-color: {ACCENT};
}}
QGroupBox {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 8px;
    color: {TEXT1};
    font-size: 10.5px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QTextEdit {{
    background: {BG2};
    color: {TEXT1};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
"""

TOOLBAR_BTN_STYLE = f"""
QPushButton {{
    background: transparent;
    color: {TEXT1};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
}}
QPushButton:hover {{
    background: {BG3};
    border-color: {BORDER2};
    color: {TEXT0};
}}
QPushButton:checked {{
    background: {ACCENT_DIM};
    border-color: {ACCENT};
    color: {ACCENT};
}}
"""

PIPELINE_LIST_STYLE = f"""
QListWidget {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT0};
    font-size: 10px;
}}
QListWidget::item {{
    padding: 5px 7px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget::item:selected {{
    background: {ACCENT_DIM};
    color: {ACCENT};
}}
"""

PIPELINE_UNDO_STYLE = f"""
QPushButton {{
    background: {BG3};
    color: {TEXT1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 10px;
}}
QPushButton:hover {{
    background: #1e0f0f;
    border-color: #4a1515;
    color: {RED};
}}
"""

METADATA_TEXT_STYLE = f"""
QTextEdit {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT0};
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}
"""