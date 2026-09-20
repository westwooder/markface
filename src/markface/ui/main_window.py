"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ..detect.runtime import available_devices, has_gpu
from ..paths import log_file, models_dir
from ..pipeline import Pipeline
from ..settings import Settings
from ..video import VideoInfo, probe
from .widgets import ImageView, LabeledSlider
from .worker import PreviewWorker, ProcessWorker, start_worker, wait_for_idle

log = logging.getLogger(__name__)

VIDEO_FILTER = ("视频文件 (*.mp4 *.mov *.avi *.mkv *.flv *.wmv *.m4v *.mpg *.mpeg "
                "*.ts *.webm);;所有文件 (*.*)")

STYLE = """
QWidget { background:#15171b; color:#d6dae0; font-size:13px; }
QGroupBox { border:1px solid #2a2e36; border-radius:8px; margin-top:14px;
            padding:12px 10px 10px 10px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px;
                   color:#9aa3ae; }
QPushButton { background:#262a32; border:1px solid #343a44; border-radius:6px;
              padding:7px 16px; }
QPushButton:hover { background:#2f343d; }
QPushButton:disabled { color:#5d636c; background:#1d2026; }
QPushButton#primary { background:#2f6feb; border-color:#2f6feb; color:#fff;
                      font-weight:600; padding:9px 22px; }
QPushButton#primary:hover { background:#3b7ef7; }
QPushButton#primary:disabled { background:#26303f; color:#6b7584;
                               border-color:#26303f; }
QLineEdit, QComboBox, QSpinBox { background:#1b1e24; border:1px solid #2f343d;
                                 border-radius:5px; padding:6px 8px; }
QComboBox::drop-down { border:none; width:18px; }
QComboBox QAbstractItemView { background:#1b1e24; selection-background-color:#2f6feb; }
QProgressBar { background:#1b1e24; border:1px solid #2f343d; border-radius:5px;
               height:20px; text-align:center; color:#c9ced6; }
QProgressBar::chunk { background:#2f6feb; border-radius:4px; }
QSlider::groove:horizontal { height:4px; background:#2f343d; border-radius:2px; }
QSlider::handle:horizontal { width:14px; height:14px; margin:-6px 0;
                             background:#5b8def; border-radius:7px; }
QSlider::sub-page:horizontal { background:#3b6fd0; border-radius:2px; }
QTabWidget::pane { border:1px solid #2a2e36; border-radius:8px; top:-1px; }
QTabBar::tab { background:#1b1e24; padding:8px 18px; border:1px solid #2a2e36;
               border-bottom:none; border-top-left-radius:6px;
               border-top-right-radius:6px; margin-right:2px; }
QTabBar::tab:selected { background:#262a32; color:#fff; }
QCheckBox::indicator { width:15px; height:15px; border:1px solid #3a4150;
                       border-radius:3px; background:#1b1e24; }
QCheckBox::indicator:checked { background:#2f6feb; border-color:#2f6feb; }
QLabel#hint { color:#7f868f; font-size:12px; }
QLabel#status { color:#9aa3ae; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.st = Settings.load()
        self.src_path: str = ""
        self.info: VideoInfo | None = None
        self.proc_thread: QThread | None = None
        self.proc_worker: ProcessWorker | None = None
        self.prev_thread: QThread | None = None
        self.prev_worker: PreviewWorker | None = None
        self._preview_pending = False
        self._last_out = ""

        self.setWindowTitle("markface · 视频人脸打码")
        self.resize(1180, 760)
        self.setMinimumSize(980, 660)
        self.setStyleSheet(STYLE)
        self.setAcceptDrops(True)

        self._build_ui()
        self._load_settings_into_ui()

        # Coalesce slider drags into one preview refresh.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(280)
        self._preview_timer.timeout.connect(self._do_preview)

    # ---------- construction ----------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        outer.addWidget(self._build_source_row())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_preview_panel())
        split.addWidget(self._build_settings_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 460])
        outer.addWidget(split, 1)

        outer.addWidget(self._build_action_row())

    def _build_source_row(self) -> QWidget:
        box = QGroupBox("① 选择视频")
        grid = QGridLayout(box)
        grid.setSpacing(8)

        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("把视频拖进窗口，或点右侧按钮选择…")
        self.src_edit.setReadOnly(True)
        btn_src = QPushButton("浏览…")
        btn_src.clicked.connect(self.choose_source)

        self.dst_edit = QLineEdit()
        self.dst_edit.setPlaceholderText("选择视频后自动填写")
        btn_dst = QPushButton("另存为…")
        btn_dst.clicked.connect(self.choose_dest)

        self.info_label = QLabel("未选择视频")
        self.info_label.setObjectName("hint")

        grid.addWidget(QLabel("输入"), 0, 0)
        grid.addWidget(self.src_edit, 0, 1)
        grid.addWidget(btn_src, 0, 2)
        grid.addWidget(QLabel("输出"), 1, 0)
        grid.addWidget(self.dst_edit, 1, 1)
        grid.addWidget(btn_dst, 1, 2)
        grid.addWidget(self.info_label, 2, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        return box

    def _build_preview_panel(self) -> QWidget:
        box = QGroupBox("② 预览效果")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        self.view = ImageView("选择视频后点「预览当前帧」查看打码效果")
        lay.addWidget(self.view, 1)

        self.frame_slider = LabeledSlider("预览位置", 0.0, 1.0, 0.3, 0.01,
                                          suffix="", decimals=2)
        self.frame_slider.valueChanged.connect(lambda _: self.schedule_preview())
        lay.addWidget(self.frame_slider)

        row = QHBoxLayout()
        self.btn_preview = QPushButton("预览当前帧")
        self.btn_preview.clicked.connect(self.schedule_preview)
        self.btn_preview.setEnabled(False)
        self.chk_boxes = QCheckBox("显示检测框（调参用）")
        self.chk_boxes.toggled.connect(self._on_boxes_toggled)
        row.addWidget(self.btn_preview)
        row.addWidget(self.chk_boxes)
        row.addStretch(1)
        lay.addLayout(row)

        self.preview_hint = QLabel("提示：滑动上面的进度条可以挑不同画面试效果")
        self.preview_hint.setObjectName("hint")
        self.preview_hint.setWordWrap(True)
        lay.addWidget(self.preview_hint)
        return box

    def _build_settings_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._tab_basic(), "常用")
        tabs.addTab(self._tab_detect(), "检测与跟踪")
        tabs.addTab(self._tab_output(), "输出")
        return tabs

    def _tab_basic(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        dev_box = QGroupBox("计算设备")
        dev_form = QFormLayout(dev_box)
        self.cmb_device = QComboBox()
        for value, label in available_devices():
            self.cmb_device.addItem(label, value)
        self.cmb_device.currentIndexChanged.connect(self._on_device_changed)
        dev_form.addRow("加速方式", self.cmb_device)
        self.lbl_device = QLabel(
            "检测到 GPU，可用 DirectML 加速" if has_gpu()
            else "未检测到可用 GPU，将使用 CPU"
        )
        self.lbl_device.setObjectName("hint")
        self.lbl_device.setWordWrap(True)
        dev_form.addRow("", self.lbl_device)
        lay.addWidget(dev_box)

        q_box = QGroupBox("识别强度")
        q_form = QFormLayout(q_box)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItem("快速（小模型，速度优先）", "fast")
        self.cmb_preset.addItem("均衡（推荐）", "balanced")
        self.cmb_preset.addItem("严格（大图输入，宁误杀不放过）", "thorough")
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        q_form.addRow("预设", self.cmb_preset)
        self.lbl_preset = QLabel("")
        self.lbl_preset.setObjectName("hint")
        self.lbl_preset.setWordWrap(True)
        q_form.addRow("", self.lbl_preset)
        lay.addWidget(q_box)

        m_box = QGroupBox("马赛克范围")
        m_lay = QVBoxLayout(m_box)
        self.sld_scale_x = LabeledSlider("横向范围", 0.4, 2.0, self.st.scale_x, 0.02, "×")
        self.sld_scale_y = LabeledSlider("纵向范围", 0.4, 2.0, self.st.scale_y, 0.02, "×")
        self.sld_offset_y = LabeledSlider("上下位置", -0.5, 0.5, self.st.offset_y, 0.01, "")
        self.sld_block = LabeledSlider("马赛克粗细", 0.02, 0.2, self.st.block_ratio, 0.005, "", 3)
        self.sld_strength = LabeledSlider("遮挡厚度", 0.5, 3.0, self.st.strength, 0.05, "×")
        self.sld_darken = LabeledSlider("压暗程度", 0.0, 0.4, self.st.darken, 0.02, "")
        self.sld_blur = LabeledSlider("柔化程度", 0.0, 0.8, self.st.blur_mix, 0.02, "")
        self.sld_feather = LabeledSlider("边缘羽化", 0.0, 30.0, float(self.st.feather), 1.0, "px", 0)
        for s in (self.sld_scale_x, self.sld_scale_y, self.sld_offset_y,
                  self.sld_block, self.sld_strength, self.sld_darken,
                  self.sld_blur, self.sld_feather):
            s.valueChanged.connect(lambda _: self.schedule_preview())
            m_lay.addWidget(s)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("形状"))
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItem("椭圆（贴合脸型，推荐）", "ellipse")
        self.cmb_shape.addItem("矩形", "rect")
        self.cmb_shape.currentIndexChanged.connect(lambda _: self.schedule_preview())
        shape_row.addWidget(self.cmb_shape, 1)
        m_lay.addLayout(shape_row)

        self.chk_flatten = QCheckBox("压平细节（避免五官轮廓透过色块）")
        self.chk_flatten.toggled.connect(lambda _: self.schedule_preview())
        m_lay.addWidget(self.chk_flatten)

        hint = QLabel("「上下位置」负值往上移、正值往下移。眼睛露出来就把它调更负，"
                      "或把纵向范围调大 —— 椭圆在左右两侧比中间矮，眼角容易漏在外面。"
                      "觉得码不够厚就调「遮挡厚度」，它会同时放大格子并压平细节；"
                      "「马赛克粗细」是格子的基准大小。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        m_lay.addWidget(hint)
        lay.addWidget(m_box)
        lay.addStretch(1)
        return w

    def _tab_detect(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        d_box = QGroupBox("检测器")
        d_lay = QVBoxLayout(d_box)
        self.sld_conf = LabeledSlider("人脸阈值", 0.05, 0.8, self.st.conf_face, 0.01, "")
        self.sld_conf.valueChanged.connect(lambda _: self.schedule_preview())
        d_lay.addWidget(self.sld_conf)
        self.chk_yunet = QCheckBox("启用 YuNet 二次检测（补漏，几乎不耗时）")
        self.chk_pose = QCheckBox("启用姿态推断头部（脸被完全遮住时靠身体定位）")
        self.chk_repair = QCheckBox("遮挡补偿（框被遮挡缩小时自动补全）")
        self.chk_reject = QCheckBox("过滤误检（去掉手臂、肩膀等平坦皮肤的误判）")
        for c in (self.chk_yunet, self.chk_pose, self.chk_repair, self.chk_reject):
            c.toggled.connect(lambda _: self.schedule_preview())
            d_lay.addWidget(c)
        conf_hint = QLabel("阈值越低找得越多、也越容易误判。蒙眼、堵嘴这类情况建议"
                           "保持全开。「过滤误检」只会去掉置信度低且完全没有"
                           "五官纹理的框，不影响正常人脸。")
        conf_hint.setObjectName("hint")
        conf_hint.setWordWrap(True)
        d_lay.addWidget(conf_hint)
        lay.addWidget(d_box)

        t_box = QGroupBox("跟踪与补帧")
        t_form = QFormLayout(t_box)
        self.chk_track = QCheckBox("启用跟踪")
        t_form.addRow("", self.chk_track)
        self.spn_max_age = QSpinBox()
        self.spn_max_age.setRange(1, 300)
        self.spn_max_age.setSuffix(" 帧")
        t_form.addRow("丢失后续遮", self.spn_max_age)
        self.spn_fill = QSpinBox()
        self.spn_fill.setRange(0, 120)
        self.spn_fill.setSuffix(" 帧")
        t_form.addRow("断档插补", self.spn_fill)
        self.spn_smooth = QSpinBox()
        self.spn_smooth.setRange(1, 21)
        self.spn_smooth.setSingleStep(2)
        self.spn_smooth.setSuffix(" 帧")
        t_form.addRow("平滑窗口", self.spn_smooth)
        self.spn_every = QSpinBox()
        self.spn_every.setRange(1, 10)
        self.spn_every.setPrefix("每 ")
        self.spn_every.setSuffix(" 帧检测一次")
        t_form.addRow("检测间隔", self.spn_every)
        t_hint = QLabel("跟踪默认关闭。<b>如果发现有漏帧（某些画面没打上码），"
                        "建议打开跟踪</b> —— 它会在人脸暂时检测不到时按运动趋势"
                        "继续打码，「断档插补」还会在整段分析完后回填中间漏掉的帧。"
                        "代价是偶尔会在人已经走开的位置多留几帧码。检测间隔设为 1 最稳。")
        t_hint.setTextFormat(Qt.TextFormat.RichText)
        t_hint.setObjectName("hint")
        t_hint.setWordWrap(True)
        t_form.addRow("", t_hint)
        lay.addWidget(t_box)
        lay.addStretch(1)
        return w

    def _tab_output(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.chk_audio = QCheckBox("保留原视频声音")
        form.addRow("", self.chk_audio)

        self.cmb_quality = QComboBox()
        self.cmb_quality.addItem("跟随原视频（推荐）", "match")
        self.cmb_quality.addItem("指定 CRF", "crf")
        self.cmb_quality.addItem("近无损（文件很大）", "lossless")
        self.cmb_quality.currentIndexChanged.connect(self._on_quality_mode)
        form.addRow("画质", self.cmb_quality)

        self.spn_quality = QSpinBox()
        self.spn_quality.setRange(0, 51)
        form.addRow("CRF 值", self.spn_quality)
        q_hint = QLabel("「跟随原视频」会读取源文件的编码和码率，按同样的规格输出，"
                        "画面观感与原片一致。音轨直接复制，不重新编码。")
        q_hint.setObjectName("hint")
        q_hint.setWordWrap(True)
        form.addRow("", q_hint)

        self.spn_threads = QSpinBox()
        self.spn_threads.setRange(0, 64)
        self.spn_threads.setSpecialValueText("自动")
        form.addRow("CPU 线程数", self.spn_threads)

        btn_log = QPushButton("打开日志目录")
        btn_log.clicked.connect(self._open_log_dir)
        form.addRow("", btn_log)

        self.lbl_models = QLabel(f"模型目录：{models_dir()}")
        self.lbl_models.setObjectName("hint")
        self.lbl_models.setWordWrap(True)
        form.addRow("", self.lbl_models)
        return w

    def _build_action_row(self) -> QWidget:
        frame = QFrame()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("就绪")
        lay.addWidget(self.progress)

        row = QHBoxLayout()
        self.status = QLabel("请选择一个视频文件")
        self.status.setObjectName("status")
        row.addWidget(self.status, 1)

        self.btn_start = QPushButton("开始打码")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        self.btn_open = QPushButton("打开输出目录")
        self.btn_open.clicked.connect(self._open_output_dir)
        self.btn_open.setEnabled(False)
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_start)
        lay.addLayout(row)
        return frame

    # ---------- settings <-> widgets ----------

    def _load_settings_into_ui(self) -> None:
        st = self.st
        i = self.cmb_device.findData(st.device)
        self.cmb_device.setCurrentIndex(i if i >= 0 else 0)
        i = self.cmb_preset.findData(st.preset)
        self.cmb_preset.setCurrentIndex(i if i >= 0 else 1)
        i = self.cmb_shape.findData(st.shape)
        self.cmb_shape.setCurrentIndex(i if i >= 0 else 0)
        i = self.cmb_quality.findData(st.quality_mode)
        self.cmb_quality.setCurrentIndex(i if i >= 0 else 0)
        self._on_quality_mode()

        self.sld_scale_x.set_value(st.scale_x)
        self.sld_scale_y.set_value(st.scale_y)
        self.sld_offset_y.set_value(st.offset_y)
        self.sld_block.set_value(st.block_ratio)
        self.sld_strength.set_value(st.strength)
        self.sld_darken.set_value(st.darken)
        self.sld_blur.set_value(st.blur_mix)
        self.sld_feather.set_value(float(st.feather))
        self.sld_conf.set_value(st.conf_face)

        self.chk_yunet.setChecked(st.use_yunet)
        self.chk_pose.setChecked(st.use_pose)
        self.chk_repair.setChecked(st.repair_occlusion)
        self.chk_flatten.setChecked(st.flatten)
        self.chk_reject.setChecked(st.reject_flat)
        self.chk_track.setChecked(st.track_enabled)
        self.chk_audio.setChecked(st.keep_audio)
        self.chk_boxes.setChecked(st.preview_boxes)

        self.spn_max_age.setValue(st.track_max_age)
        self.spn_fill.setValue(st.forward_fill)
        self.spn_smooth.setValue(st.smooth_window)
        self.spn_every.setValue(st.detect_every)
        self.spn_quality.setValue(st.quality)
        self.spn_threads.setValue(st.threads)
        self._update_preset_hint()

    def collect_settings(self) -> Settings:
        st = self.st
        st.device = self.cmb_device.currentData() or "auto"
        st.preset = self.cmb_preset.currentData() or "balanced"
        st.shape = self.cmb_shape.currentData() or "ellipse"
        st.scale_x = self.sld_scale_x.value()
        st.scale_y = self.sld_scale_y.value()
        st.offset_y = self.sld_offset_y.value()
        st.block_ratio = self.sld_block.value()
        st.strength = self.sld_strength.value()
        st.darken = self.sld_darken.value()
        st.blur_mix = self.sld_blur.value()
        st.feather = int(round(self.sld_feather.value()))
        st.conf_face = self.sld_conf.value()
        st.use_yunet = self.chk_yunet.isChecked()
        st.use_pose = self.chk_pose.isChecked()
        st.repair_occlusion = self.chk_repair.isChecked()
        st.flatten = self.chk_flatten.isChecked()
        st.reject_flat = self.chk_reject.isChecked()
        st.track_enabled = self.chk_track.isChecked()
        st.keep_audio = self.chk_audio.isChecked()
        st.preview_boxes = self.chk_boxes.isChecked()
        st.track_max_age = self.spn_max_age.value()
        st.forward_fill = self.spn_fill.value()
        st.smooth_window = self.spn_smooth.value()
        st.detect_every = self.spn_every.value()
        st.quality = self.spn_quality.value()
        st.quality_mode = self.cmb_quality.currentData() or "match"
        st.threads = self.spn_threads.value()
        return st.clamp()

    def _update_preset_hint(self) -> None:
        key = self.cmb_preset.currentData()
        text = {
            "fast": "只用小模型，速度最快；遮挡场景可能漏。",
            "balanced": "中等模型逐帧检测，日常够用，误检少。",
            "thorough": "960 输入 + 低阈值，并开启 YuNet、姿态推断、遮挡补偿；"
                        "蒙眼堵嘴这类重遮挡用这档，速度约慢一半。",
        }.get(key, "")
        self.lbl_preset.setText(text)

    @Slot()
    def _on_preset_changed(self) -> None:
        from ..settings import PRESETS
        key = self.cmb_preset.currentData()
        p = PRESETS.get(key, {})
        # A preset is a starting point: push its detector choices into the
        # checkboxes so the user can see and then override them.
        self.sld_conf.set_value(p.get("conf_face", self.st.conf_face))
        self.chk_yunet.setChecked(bool(p.get("use_yunet", False)))
        self.chk_pose.setChecked(bool(p.get("use_pose", False)))
        self.chk_repair.setChecked(bool(p.get("repair_occlusion", False)))
        self.spn_every.setValue(int(p.get("detect_every", 1)))
        self._update_preset_hint()
        self._invalidate_ensemble()
        self.schedule_preview()

    @Slot()
    def _on_quality_mode(self) -> None:
        self.spn_quality.setEnabled(self.cmb_quality.currentData() == "crf")

    @Slot()
    def _on_device_changed(self) -> None:
        self._invalidate_ensemble()
        self.schedule_preview()

    @Slot(bool)
    def _on_boxes_toggled(self, _checked: bool) -> None:
        self.schedule_preview()

    def _invalidate_ensemble(self) -> None:
        """Force the next preview to rebuild sessions (model/device changed)."""
        self._preview_pending = False

    # ---------- file selection ----------

    @Slot()
    def choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if path:
            self.load_source(path)

    def load_source(self, path: str) -> None:
        info = probe(path)
        if info is None:
            QMessageBox.warning(self, "无法打开", f"这个文件读不出视频帧：\n{path}")
            return
        self.src_path = path
        self.info = info
        self.src_edit.setText(path)
        self.info_label.setText(info.label())

        p = Path(path)
        self.dst_edit.setText(str(p.with_name(f"{p.stem}_马赛克.mp4")))
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.status.setText("已载入视频，可以先预览，也可以直接开始")
        self.progress.setValue(0)
        self.progress.setFormat("就绪")
        self.schedule_preview()

    @Slot()
    def choose_dest(self) -> None:
        start = self.dst_edit.text() or ""
        path, _ = QFileDialog.getSaveFileName(self, "保存为", start,
                                              "MP4 视频 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.dst_edit.setText(path)

    # Drag and drop straight onto the window.
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p:
                self.load_source(p)
                break

    # ---------- preview ----------

    @Slot()
    def schedule_preview(self) -> None:
        if not self.src_path:
            return
        self._preview_timer.start()

    def _do_preview(self) -> None:
        if not self.src_path or self.info is None:
            return
        if self.proc_thread is not None and self.proc_thread.isRunning():
            return
        if self.prev_thread is not None and self.prev_thread.isRunning():
            # A preview is already in flight; ask for another once it lands.
            self._preview_pending = True
            return
        st = self.collect_settings()
        idx = int(self.frame_slider.value() * max(0, self.info.frames - 1))
        self.preview_hint.setText("正在生成预览…")

        self.prev_worker = PreviewWorker(self.src_path, idx, st)
        self.prev_worker.ready.connect(self._on_preview_ready)
        self.prev_worker.failed.connect(self._on_preview_failed)
        self.prev_thread = start_worker(self.prev_worker)

    @Slot(object, int, str)
    def _on_preview_ready(self, frame, count: int, desc: str) -> None:
        self.view.set_frame(frame)
        self.preview_hint.setText(f"这一帧找到 {count} 个目标 · {desc}")
        self._after_preview()

    @Slot(str)
    def _on_preview_failed(self, msg: str) -> None:
        self.preview_hint.setText(f"预览失败：{msg}")
        self._after_preview()

    def _after_preview(self) -> None:
        if self._preview_pending:
            self._preview_pending = False
            QTimer.singleShot(60, self._do_preview)

    # ---------- processing ----------

    @Slot()
    def start_processing(self) -> None:
        if not self.src_path or self.info is None:
            return
        dst = self.dst_edit.text().strip()
        if not dst:
            QMessageBox.information(self, "缺少输出路径", "请先指定输出文件。")
            return
        if Path(dst).resolve() == Path(self.src_path).resolve():
            QMessageBox.warning(self, "路径冲突", "输出文件不能和输入文件相同。")
            return
        if Path(dst).exists():
            r = QMessageBox.question(
                self, "文件已存在", f"{dst}\n已存在，要覆盖吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return

        st = self.collect_settings()
        st.save()

        self._set_busy(True)
        self.progress.setValue(0)
        self.progress.setFormat("准备中…")
        self.status.setText("正在初始化模型…")

        self.proc_worker = ProcessWorker(self.src_path, dst, st)
        self.proc_worker.progress.connect(self._on_progress)
        self.proc_worker.finished.connect(self._on_finished)
        self.proc_thread = start_worker(self.proc_worker)

    @Slot()
    def cancel_processing(self) -> None:
        if self.proc_worker:
            self.proc_worker.cancel()
            self.status.setText("正在取消…")
            self.btn_cancel.setEnabled(False)

    @Slot(str, float, str)
    def _on_progress(self, stage: str, frac: float, msg: str) -> None:
        # Detection is pass 1, rendering is pass 2: map each onto half the bar.
        base = {"probe": 0.0, "detect": 0.0, "render": 0.5}.get(stage, 0.0)
        span = {"probe": 0.02, "detect": 0.5, "render": 0.5}.get(stage, 0.0)
        overall = base + span * frac
        self.progress.setValue(int(overall * 1000))
        label = {"probe": "读取", "detect": "第 1/2 步 分析", "render": "第 2/2 步 输出"}
        self.progress.setFormat(f"{label.get(stage, stage)} {overall * 100:.0f}%")
        self.status.setText(msg)

    @Slot(object)
    def _on_finished(self, res) -> None:
        self._set_busy(False)
        if res.ok:
            self.progress.setValue(1000)
            self.progress.setFormat("完成")
            self.status.setText(res.message)
            self.btn_open.setEnabled(True)
            self._last_out = res.out_path
            QMessageBox.information(self, "完成", f"{res.message}\n\n输出：{res.out_path}")
        else:
            self.progress.setFormat("未完成")
            self.status.setText(res.message or "处理失败")
            if res.message and res.message != "已取消":
                QMessageBox.warning(self, "处理失败", res.message)

    def _set_busy(self, busy: bool) -> None:
        self.btn_start.setEnabled(not busy and bool(self.src_path))
        self.btn_cancel.setEnabled(busy)
        self.btn_preview.setEnabled(not busy and bool(self.src_path))
        for w in (self.cmb_device, self.cmb_preset, self.spn_every):
            w.setEnabled(not busy)

    def _open_output_dir(self) -> None:
        target = getattr(self, "_last_out", "") or self.dst_edit.text()
        if target:
            self._reveal(Path(target).parent)

    def _open_log_dir(self) -> None:
        self._reveal(log_file().parent)

    @staticmethod
    def _reveal(folder: Path) -> None:
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.proc_worker and self.proc_thread and self.proc_thread.isRunning():
            r = QMessageBox.question(
                self, "正在处理", "处理还没完成，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.proc_worker.cancel()
            wait_for_idle(8000)
        self._preview_timer.stop()
        if self.prev_thread is not None and self.prev_thread.isRunning():
            wait_for_idle(5000)
        try:
            self.collect_settings().save()
        except Exception:  # noqa: BLE001
            pass
        event.accept()
