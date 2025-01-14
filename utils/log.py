import functools
import logging
from pathlib import Path
import shutil
import glob

from datetime import datetime, timedelta

_LOG_DICT = {}

@functools.lru_cache(None)  # always the same :)
def get_datestring_for_the_run():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _make_key(msg, args, kwargs):
    args_str = ', '.join([str(arg) for arg in args])
    kwargs_str = ', '.join([f'{str(k)}={str(v)}' for k, v in kwargs.items()])
    r = [msg]
    if args_str or kwargs_str:
        r.append(' % (')
    r.append(args_str)
    if args_str:
        r.append(', ')
    r.append(kwargs_str)
    if args_str or kwargs_str:
        r.append(')')
    # MyMessage % (arg1, arg2, kw1=v1m, kw2=v2m)
    return ''.join(r)


def debug_once(msg, *args, logger=None, **kwargs):
    key = _make_key(msg, args, kwargs)

    lvl = logging.DEBUG
    t = datetime.now()
    should_log = True

    if key in _LOG_DICT:
        plvl, pt = _LOG_DICT[key]
        # Do not overwrite
        if plvl > lvl:
            t = pt
            lvl = plvl
        should_log = False

    _LOG_DICT[key] = (lvl, t)
    if should_log:
        logger.debug(msg, *args, **kwargs)
    return should_log


def info_once(msg, *args, logger=None, **kwargs):
    key = _make_key(msg, args, kwargs)

    lvl = logging.INFO
    t = datetime.now()
    should_log = True

    if key in _LOG_DICT:
        plvl, pt = _LOG_DICT[key]
        should_log = plvl <= lvl and t - pt > timedelta(minutes=5)
        lvl = max(lvl, plvl)

    _LOG_DICT[key] = (lvl, t)
    if should_log:
        logger.info(msg, *args, **kwargs)
    return should_log


def getLogger(name):
    if name != 'lrtl' and not name.startswith('lrtl.'):
        name = 'lrtl.' + name
    logger = logging.getLogger(name)
    logger.info_once = functools.partial(info_once, logger=logger)
    logger.debug_once = functools.partial(info_once, logger=logger)
    return logger


def checkpoint_code(log_path):
    code_path = Path(log_path) / 'code'
    if code_path.exists():
        code_path = code_path.with_name(f'code_{get_datestring_for_the_run()}')
    code_path.mkdir(parents=True, exist_ok=True)
    for file in glob.glob('*.py'):
        shutil.copy(file, code_path)
    shutil.copytree('datasets', code_path / 'datasets', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('losses', code_path / 'losses', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('utils', code_path / 'utils', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('mask_former', code_path / 'mask_former', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))


class log_level:
    def __init__(self, logger, lvl=logging.INFO):
        self.logger = logging.getLogger(logger)
        self.lvl = lvl
        self.current_lvl = self.logger.level

    def __enter__(self):
        self.current_lvl = self.logger.level
        self.logger.setLevel(self.lvl)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.current_lvl)
