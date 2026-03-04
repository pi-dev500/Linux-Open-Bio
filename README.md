# Complete Auth for Linux
  ⚠️ The only way to configure is editing the headers of the scripts. Maybe a conffile will be a future feature

  ⚠️ As every face recognition tools, its precision is not perfect, recommended similarity threshold is more than 0.6 (I use 0.7 by default and the similarity indice with my friend on my bad HP laptop camera is between 0 and 0.3). I cannot be taken responsible for any theft / data loss due to the use of this software.

  It works very good on both Intel and AMD CPUs, support for NPU only Intel yet due to no cross-platform npu+cpu auto driver.
## Dependencies

### Fedora / RHEL / CentOS

```bash
sudo dnf install -y glib2-devel libfprint-devel polkit-devel dbus-devel systemd-devel pam-devel gcc pkg-config meson ninja-build libxslt-devel python3 python3-devel python3-pip
pip3 install -r requirements.txt
```

### Arch Linux

```bash
sudo pacman -S --noconfirm glib2 libfprint polkit dbus systemd linux-pam gcc pkg-config meson ninja libxslt python python-pip
pip install -r requirements.txt
```

### Debian / Ubuntu / Linux Mint

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-dev libfprint-2-dev libpolkit-gobject-1-dev libdbus-1-dev libsystemd-dev libpam0g-dev gcc pkg-config meson ninja-build libxslt1-dev python3 python3-dev python3-pip
pip3 install -r requirements.txt
```

### openSUSE / SUSE

```bash
sudo zypper install -y glib2-devel libfprint-devel polkit-devel dbus-1-devel systemd-devel pam-devel gcc pkg-config meson ninja libxslt-devel python3 python3-devel python3-pip
pip3 install -r requirements.txt
```

### Dependency Summary

#### System Libraries
- **GLib 2.56+** — Core system libraries (glib2-devel, gio, gio-unix, gmodule)
- **libfprint 1.94.0+** — Fingerprint reader support
- **PolKit 0.91+** — Authorization framework (polkit-devel)
- **D-Bus** — Inter-process communication
- **libsystemd** — Systemd integration (systemd 235+)
- **PAM** — Pluggable Authentication Modules (libpam-devel)
- **OpenVINO** — AI inference engine for face recognition
- **Build tools** — gcc, pkg-config, meson, ninja, libxslt

#### Python Packages
- `pydbus` — D-Bus Python bindings
- `PyGObject` — GObject Python bindings
- `opencv-python` — Computer vision library
- `numpy` — Numerical computing
- `openvino` — OpenVINO Python API
- `pyyaml` — YAML parsing
- `requests` — HTTP library

## Installation
### Recommended:
- Run `./install.sh`, Should automatically install anything in place.
### Alternate:
- First, run download_models.py to download the needed models fore the face identification to work.
- Then, run compile.sh to build and install the PAM module on the system (don't worry, this module is not enabled in PAM config by default, and don't do anything if the daemon doesn't run.)
- Create the host directory: `sudo mkdir /usr/share/face_recognition`
  You can use another path, but you'll need to edit the system module for on-boot autostart.
- Copy the content of the downloaded repo to the new system directory: `sudo cp ./* /usr/share/face_recognition` if you are in the github dir.
- Move system service and dbus config system-wide: `sudo mv /usr/share/face_recognition/etc /`

## Use

- The daemon must be ran as root. You can enable the provided service file for on-boot startup.
- add the line:
  ```
  auth sufficient pam_dual_grosshack.so
  ```
  At the beginning (or the end) of the PAM configs files (in `/etc/pam.d`) that you want to allow face recognition.
- To add a new face fo current user
  ```
  /usr/share/face_recognition/facerec.py add [capture path] [face name]
  ```
- To check if the user who is currently facing the computer is the logged in user:
  ```
  /usr/share/face_recognition/facerec.py check
  ```
- To remove all faces for the logged user:
  ```
  /usr/share/face_recognition/facerec.py remove [all]
  ```
- To remove given face names for the user:
  ```
  /usr/share/face_recognition/facerec.py remove (face1) (face2) ...
  ```
## Features:
- Ultra-fast recognition: Utilizes light & accurate OpenVINO models for rapid face detection and identification on any x86_64 processor.

- Low resource usage: Optimized for minimal CPU and memory consumption.

- Camera fallback: Automatically switches between cameras following the configured fallback order.

- Image quality checks: Filters out blurry, under-lit, or over-lit images before recognition for more safety.

- Anti-Spoof: Use accurate miniFASnetv2 anti-spoofing CNN.

- Multi-user support: Stores separate embedding dictionaries per Linux user, allowing multiple profiles.
  
- It is not possible to retrieve the photos of the face from the storage of embeddings: Only vectors are stored.
