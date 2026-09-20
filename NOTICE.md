# 第三方组件声明

markface 本身以 AGPL-3.0 授权（见 [LICENSE](LICENSE)）。

选择 AGPL 不是偏好，而是义务：项目使用的人脸与姿态检测模型派生自 Ultralytics
YOLO（AGPL-3.0）和 akanametov/yolo-face（GPL-3.0），这两个许可具有传染性，
分发衍生作品时整体必须采用同等或兼容的许可。

## 模型

| 模型 | 用途 | 来源 | 许可 |
|---|---|---|---|
| yolov11s-face / yolov11m-face | 人脸检测 | [deepghs/yolo-face](https://huggingface.co/deepghs/yolo-face)，导出自 [akanametov/yolo-face](https://github.com/akanametov/yolo-face) | **GPL-3.0** |
| yolo26n-pose | 姿态推断头部 | [onnx-community/yolo26n-pose-ONNX](https://huggingface.co/onnx-community/yolo26n-pose-ONNX)，基于 [Ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** |
| yunet | 人脸二次检测 | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT（Copyright © 2020 Shiqi Yu） |

模型权重的 SHA256 记录在 [models/CHECKSUMS.txt](models/CHECKSUMS.txt)。

## 运行库

| 组件 | 许可 | 说明 |
|---|---|---|
| [onnxruntime-directml](https://github.com/microsoft/onnxruntime) | MIT | 模型推理 |
| [PySide6](https://www.qt.io/qt-for-python) | LGPL-3.0 | 界面。以动态链接方式使用，未修改 Qt 本身 |
| [OpenCV](https://opencv.org/) | Apache-2.0 | 图像处理、视频解码 |
| [NumPy](https://numpy.org/) | BSD-3-Clause | 数值计算 |
| [FFmpeg](https://ffmpeg.org/) | **GPL-3.0**（发布包内为 GPL 构建） | 视频编码、音轨复制 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | BSD-2-Clause | 定位 ffmpeg 可执行文件 |
| CPython | PSF-2.0 | 打包在发布版内的运行环境 |

## 关于 LGPL 与 PySide6

LGPL-3.0 要求使用者能够替换库本身。发布包采用 PyInstaller onedir 形式，
Qt 的 DLL 以独立文件存在于 `_internal\` 目录中，可直接替换为同版本的其他构建，
符合 LGPL 的要求。Qt 源码可从 [download.qt.io](https://download.qt.io/) 获取。

## 如果你要复用这份代码

AGPL-3.0 的核心义务：

- 分发程序（含二进制）时必须提供完整源码，或提供获取源码的途径
- 衍生作品必须同样以 AGPL-3.0 发布
- **如果通过网络提供服务**（比如做成在线打码网站），使用者有权获得该服务端的完整源码

想避开这些约束，需要把 GPL/AGPL 的模型换成许可宽松的替代品。仓库里的 YuNet（MIT）
可以单独使用：在「检测与跟踪」中关闭主模型之外的检测器并不足够 —— 主模型本身
就是 GPL 的，必须替换 `models/manifest.json` 中的条目并重新训练或另寻模型。
