# hei-mujoco-lab

这是当前验收通过的实验室静态房间配置，包含完整房间、HEI 双臂机器人、桌面上的三个彩色方块和一瓶约 500 mL 的水。配置使用 MuJoCo 3.5.0，底盘和升降保持固定，手臂使用有限力位置执行器，物体使用真实接触动力学。

仓库中的 `model/` 是可运行场景的完整资源。机器人网格与原 HEI 项目保持一致，房间 XML 已包含碰撞、摩擦、重力、相机和固定升降设置。`scripts/scene.py` 提供动作目标映射和全姿态 IK，`simulate.py` 提供观察者视角和自动隐藏遮挡墙体。

## 安装

先克隆仓库，建议使用 Python 3.10：

```sh
git clone https://github.com/YouShouldBetOnMe/hei-mujoco-lab.git
cd hei-mujoco-lab
```

使用 Conda：

```sh
conda env create -f environment.yml
conda activate hei-mujoco-lab
python check_environment.py
```

或者使用 Python 虚拟环境（macOS / Linux）：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_environment.py
```

`requirements.txt` 锁定本机已验证的 MuJoCo 3.5.0、NumPy 2.2.6 和 SciPy 1.15.2；macOS 手柄依赖 PyObjC 12.2.2。Linux 使用原生窗口时需要可用的显示环境和 OpenGL 驱动；无窗口物理检查无需图形界面。

验证脚本会加载真实 XML、推进 500 个物理步，并检查模型尺寸、重力、步长、物理警告和 37 个模型文件的 SHA-256。

## 启动

激活安装依赖的环境后，在 macOS、Linux、Windows 上均可运行：

```sh
python launch.py
```

macOS 也可以双击 [launch_scene.command](launch_scene.command)；它优先使用仓库内的 `.venv`，然后查找 PATH 中的 Python。使用 Conda 时从已激活的终端运行：

```sh
./launch_scene.command
```

Python 启动器会在 macOS 自动选择当前环境的 `mjpython`，Linux 和 Windows 使用当前 Python。如果要显式指定 macOS MuJoCo 可执行文件，可以运行：

```sh
MUJOCO_PYTHON=/path/to/mjpython ./launch_scene.command
```

也可以运行无窗口检查：

```sh
python simulate.py --headless --steps 1000
```

## 参数约定

- 重力为 `(0, 0, -9.81) m/s^2`，物理步长为 `0.002 s`。
- 模型为 `nq=48, nv=44, nu=16`，没有 equality 和 mocap。
- 三个方块为 `80 x 80 x 80 mm`、每个约 `120 g`。
- 满水瓶约 `65 mm` 直径、`220 mm` 高、约 `530 g`。
- 桌面和物体的摩擦参数为 `(0.8, 0.01, 0.001)`，机器人碰撞体为 `(1.0, 0.01, 0.001)`。
- 实验室底盘和升降为固定状态；动作接口最后四维保持为零。
- 红色双开门保留为封闭、不可通行的静态结构。
- 观察者视角会根据视线隐藏遮挡墙体；机器人相机保持完整房间视野。
- 保留本机已验收的机器人自碰撞关闭设置；物理参数是当前仿真基线，并非实机标定值。

## Xbox 手柄

macOS 连接 Xbox 手柄后双击 [手柄控制.command](手柄控制.command)：

```sh
./手柄控制.command
```

或者在已激活的环境中运行 `python launch.py --gamepad`。手柄读取目前只支持 macOS；Windows 和 Linux 可以打开同一场景。

点击 MuJoCo 窗口，松开 RT 并回中摇杆，再按住 RT 操作。A/B 选择左/右臂，左摇杆控制平面移动，右摇杆控制高度和偏航，LB/RB 控制夹爪，LT 进入精细模式，Menu 暂停恢复。实验室配置不会移动底盘和升降。

手柄读取使用 macOS 原生 GameController，不需要额外的硬件驱动。窗口失去焦点、手柄断连或输入超时会停止运动指令；重连后需要先回中和释放 RT。详细映射见 [gamepad/README.md](gamepad/README.md)。

## 仓库范围

发布库只保存可复现配置和运行代码。它不包含本机 Python 虚拟环境、缓存、原始扫描、训练数据、动作尝试、运行日志或模型权重。当前已验证的 633 帧三方块示范仍留在原工作区；需要共享数据时应单独使用数据集发布渠道。

机器人网格来源于 [lipengdong/hei-rebot-lift](https://github.com/lipengdong/hei-rebot-lift)，相关 Apache 2.0 许可文本见 [LICENSE-HEI-ROBOT.txt](LICENSE-HEI-ROBOT.txt)。
