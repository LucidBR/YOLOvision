 

import contextlib
import glob
import os
import urllib
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir, *args, **kwargs):' context manager
    def __init__(self, new_dir, *args, **kwargs):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self, *args, **kwargs):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb, *args, **kwargs):
        os.chdir(self.cwd)


def increment_path(path, exist_ok=False, sep='', mkdir=False, *args, **kwargs):

    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def file_age(path=__file__, *args, **kwargs):
    # Return days since last file update
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__, *args, **kwargs):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def file_size(path, *args, **kwargs):
    # Return file/dir size (MB)
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    return 0.0


def url2file(url, *args, **kwargs):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def get_latest_run(search_dir='.', *args, **kwargs):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''
