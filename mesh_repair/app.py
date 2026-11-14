"""Application entry point."""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    app = QApplication(argv)
    app.setApplicationName("机械网格修复工具")
    from .ui.main_window import MainWindow

    font = QFont()
    for family in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "Sans"):
        font.setFamily(family)
        break
    font.setPointSize(10)
    app.setFont(font)

    win = MainWindow()
    win.show()

    # optional files on the command line, useful for scripts and tests
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        p = Path(arg)
        if p.suffix.lower() in {".stl", ".obj", ".ply"}:
            win.load_path(str(p))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
