 
import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import urllib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import psutil
import requests
import torch
from matplotlib import font_manager

from YOLOvision.yolo.utils import (AUTOINSTALL, LOGGER, ONLINE, ROOT, USER_CONFIG_DIR, TryExcept, colorstr, downloads,
                                   emojis, is_colab, is_docker, is_kaggle, is_online, is_pip_package)


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0, *args, **kwargs):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int or List[int], *args, **kwargs): Image size.
        stride (int, *args, **kwargs): Stride value.
        min_dim (int, *args, **kwargs): Minimum number of dimensions.
        floor (int, *args, **kwargs): Minimum allowed value for image size.

    Returns:
        List[int]: Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz,int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
                        f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list " \
              "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        LOGGER.warning(f" Opps Wait  updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f'Opps Wait  imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  name: str = 'version ',
                  pinned: bool = False,
                  hard: bool = False,
                  detail: bool = False) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        detail (bool): If True, print warning message if minimum version is not met.

    Returns:
        bool: True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'Opps Wait  {name}{minimum} is required by YOLOvision, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(warning_message)  # assert min requirements met
    if detail and not result:
        LOGGER.warning(warning_message)
    return result


def check_latest_pypi_version(package_name='YOLOvision', *args, **kwargs):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        str: The latest version of the package.
    """
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    if response.status_code == 200:
        return response.json()['info']['version']
    return None


def check_font(font='Arial.ttf', *args, **kwargs):

    name = Path(font).name

    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches, *args, **kwargs):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f'https://ULC.com/assets/{name}'
    if downloads.is_url(url, *args, **kwargs):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = '3.7.0') -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.

    Returns:
        None
    """
    return check_version(platform.python_version(), minimum, name='Python ', hard=True)


@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt', exclude=(), install=True, cmds='', *args, **kwargs):
    pass

def check_suffix(file='YOLOvisionn.pt', suffix='.pt', msg='', *args, **kwargs):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'



def check_file(file, suffix='', download=True, hard=True, *args, **kwargs):
    
    check_suffix(file, suffix)  
    file = str(file).strip()  
    
    if not file or ('://' not in file and Path(file).exists()): 
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):
        url = file
        file = Path(urllib.parse.unquote(file).split('?')[0]).name
        if Path(file).exists():
            LOGGER.info(f'Found {url} locally at {file}')  
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = []
        for d in 'models', 'datasets', 'tracker/cfg', 'yolo/cfg':  
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  


def check_yaml(file, suffix=('.yaml', '.yml'), hard=True, *args, **kwargs):
 
    return check_file(file, suffix, hard=hard)


def check_imshow(warn=False, *args, **kwargs):
    # Check if environment supports image displays
    try:
        assert not any((is_colab(), is_kaggle(), is_docker()))
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'Opps Wait  Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


def check_yolo(detail=True, device='', *args, **kwargs):
    from YOLOvision.yolo.utils.torch_utils import select_device

    if is_colab():
        shutil.rmtree('sample_data', ignore_errors=True)  # remove colab /sample_data directory

    if detail:
        # System info
        gib = 1 << 30  # bytes per GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        s = f'({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)'
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display
            display.clear_output()
    else:
        s = ''

    select_device(device=device, newline=False)
    LOGGER.info(f'Setup complete ✅ {s}')


def git_describe(path=ROOT, *args, **kwargs):

    try:
        assert (Path(path) / '.git').is_dir()
        return subprocess.check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except AssertionError:
        return ''


def print_args(args: Optional[dict] = None, show_file=True, show_func=False, **kwargs):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
