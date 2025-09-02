# desktop_butler.py
# UTF-8
# Требования: watchdog, PyYAML; опционально: scikit-learn, joblib

"""
Desktop Butler — кроссплатформенный инструмент автоматической уборки рабочего
стола: разовая уборка (scan-once) и мониторинг (watch), сортировка файлов по
категориям в датированные папки Important/Unimportant, безопасное удаление
нулевых файлов, исключения/чёрный список, журнал сессии с откатом и
"атомарная" обработка директорий с опциональным ZIP‑сжатием.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import shutil
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # Чтение YAML-конфига (FullLoader), важно хранить в UTF‑8. 

from watchdog.observers import Observer  # Кроссплатформенный мониторинг FS. 
from watchdog.events import FileSystemEventHandler  # Базовый обработчик событий. 

# Опциональный ML для "важности" — Naive Bayes + joblib (рекомендация sklearn). 
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import zipfile  # Управление ZIP‑архивами и compresslevel в Python 3.7+. 


# ---------------------- Конфиг и логирование ----------------------

def expand(path_str: str) -> Path:
    """
    Разворачивает "~" в домашний каталог и возвращает абсолютный Path. 
    """
    return Path(os.path.expanduser(path_str)).resolve()


def load_config(cfg_path: Path) -> dict:
    """
    Загружает YAML‑конфиг через PyYAML FullLoader, ожидая UTF‑8 и корректные
    кавычки/отступы; используйте прямые слеши или одинарные кавычки для путей. 
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def setup_logging(log_dir: Path) -> None:
    """
    Инициализирует логирование: stdout + файл с ротацией до 2 МБ × 5 файлов.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "desktop_butler.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5,
                             encoding="utf-8")
    fh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info("Logging initialized at %s", log_file)


# ---------------------- Вспомогательные функции (файлы) ----------------------

def is_zero_byte(file_path: Path) -> bool:
    """
    Возвращает True, если файл имеет размер 0 байт; безопасно удалять.
    """
    try:
        return file_path.is_file() and file_path.stat().st_size == 0
    except FileNotFoundError:
        return False


def is_stable(file_path: Path, attempts: int, interval: float) -> bool:
    """
    Проверяет стабильность размера файла (N попыток с интервалом),
    чтобы не перемещать недописанные файлы. [2]
    """
    try:
        prev_size = None
        for _ in range(attempts):
            if not file_path.exists():
                return False
            size = file_path.stat().st_size
            if prev_size is not None and size != prev_size:
                time.sleep(interval)
                prev_size = size
                continue
            prev_size = size
            time.sleep(interval)
        return True
    except Exception:
        return False


def unique_destination(dst_dir: Path, name: str) -> Path:
    """
    Возвращает уникальный путь назначения: добавляет "(N)" при конфликте имён.
    """
    base = Path(name).stem
    ext = Path(name).suffix
    candidate = dst_dir / name
    counter = 1
    while candidate.exists():
        candidate = dst_dir / f"{base} ({counter}){ext}"
        counter += 1
    return candidate


# ---------------------- Исключения и "чёрный список" ----------------------

def matches_exclusions(p: Path, cfg: dict, cleanup_root: Path) -> bool:
    """
    Проверяет, нужно ли игнорировать путь p (исключения/чёрный список).
    """
    ex = cfg.get("exclusions", {})
    names = set(ex.get("names", []))
    patterns = ex.get("patterns", [])

    if p.name in names:
        return True

    posix_path = p.as_posix()
    if any(fnmatch.fnmatch(posix_path, pat) or fnmatch.fnmatch(p.name, pat)
           for pat in patterns):
        return True

    if ex.get("inside_cleanup_root", True):
        try:
            if cleanup_root in p.resolve().parents:
                return True
        except Exception:
            return True

    bl = cfg.get("blacklist", {})

    if p.name in set(bl.get("names", [])):
        return True

    bl_exts = {e.lower().lstrip(".") for e in bl.get("extensions", [])}
    if p.is_file() and p.suffix.lower().lstrip(".") in bl_exts:
        return True

    for raw in bl.get("exact_paths", []):
        try:
            if p.resolve() == expand(raw).resolve():
                return True
        except Exception:
            return True

    for raw in bl.get("dirs", []):
        try:
            d = expand(raw).resolve()
            pr = p.resolve()
            if hasattr(Path, "is_relative_to"):
                if pr.is_relative_to(d):
                    return True
            else:
                common = os.path.commonpath([str(pr), str(d)])
                if common == str(d):
                    return True
        except Exception:
            return True

    for pat in bl.get("patterns", []):
        if fnmatch.fnmatch(posix_path, pat) or fnmatch.fnmatch(p.name, pat):
            return True

    return False


# ---------------------- Категоризация/важность (файлы) ----------------------

def categorize(file_path: Path, cfg: dict) -> str:
    """
    Возвращает имя категории по расширению согласно cfg['categories'].
    """
    ext = file_path.suffix.lower().lstrip(".")
    for cat, exts in cfg["categories"].items():
        if ext in exts:
            return cat
    return "other"


def important_by_rules(file_path: Path, cfg: dict) -> bool:
    """
    Правила "важности" без ML: ключевые слова, порог размера и "свежесть". 
    """
    name = file_path.name.lower()
    rules = cfg.get("importance_rules", {})
    high_kw = [k.lower() for k in rules.get("high_keywords", [])]
    low_kw = [k.lower() for k in rules.get("low_keywords", [])]
    try:
        st = file_path.stat()
        if any(k in name for k in high_kw):
            return True
        if any(k in name for k in low_kw):
            return False
        if st.st_size >= int(rules.get("min_size_bytes_for_high", 200000)):
            return True
        days = int(rules.get("recent_days_for_high", 7))
        if (time.time() - st.st_mtime) <= days * 86400:
            return True
    except FileNotFoundError:
        return False
    return False


# ---------------------- ML (опционально) ----------------------

def train_importance_model(labeled_csv: Path, model_path: Path) -> None:
    """
    Обучает Naive Bayes на именах файлов (bag-of-words) и сохраняет модель.
    """
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn/joblib not available. Install requirements.")

    texts, labels = [], []
    with open(labeled_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("path,"):
                continue
            try:
                path_str, label_str = line.rsplit(",", 1)
                texts.append(Path(path_str).name.lower())
                labels.append(int(label_str))
            except ValueError:
                continue

    if len(texts) < 50:
        raise ValueError("Need at least 50 labeled rows to train a minimal model.")

    pipe: Pipeline = Pipeline([
        ("vec", CountVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)


def predict_importance_ml(name_like: str, model_path: Path) -> Optional[bool]:
    """
    Предсказывает важность по имени файла через сохранённую модель; None если
    модель отсутствует или ML отключён. [2]
    """
    if not SKLEARN_OK or not model_path.exists():
        return None
    pipe: Pipeline = joblib.load(model_path)
    proba = pipe.predict_proba([name_like.lower()])
    return bool(pipe.classes_[proba.argmax()] == 1)


# ---------------------- Сессии и карантин ----------------------

def session_file(sessions_dir: Path) -> Path:
    """
    Создаёт файл журнала сессии вида session_YYYYmmdd_HHMMSS.json. 
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    return sessions_dir / f"session_{sid}.json"


def append_session_record(sess_path: Path, records: List[Dict]) -> None:
    """
    Добавляет записи (src/dst/категория/важность/время/действие) в JSON‑журнал.
    """
    existing = []
    if sess_path.exists():
        try:
            existing = json.loads(sess_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(records)
    sess_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def get_last_session(sessions_dir: Path) -> Optional[Path]:
    """
    Возвращает путь к последней сессии или None, если журналов нет. 
    """
    if not sessions_dir.exists():
        return None
    sessions = sorted(sessions_dir.glob("session_*.json"))
    return sessions[-1] if sessions else None


def purge_quarantine(quarantine_dir: Path, days: int) -> int:
    """
    Удаляет файлы в карантине старше N дней; возвращает число удалённых. 
    """
    if not quarantine_dir.exists():
        return 0
    cutoff = time.time() - days * 86400
    count = 0
    for root, _, files in os.walk(quarantine_dir):
        for fname in files:
            p = Path(root) / fname
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    count += 1
            except FileNotFoundError:
                pass
    return count


# ---------------------- Контекст исполнения ----------------------

@dataclass
class CleanerContext:
    """
    Контекст обработки: конфиг, пути, флаги, параметры стабилизации и журнал.
    """
    cfg: dict
    desktop: Path
    cleanup_root: Path
    quarantine_dir: Path
    dry_run: bool
    stable_attempts: int
    stable_interval: float
    model_path: Path
    use_ml: bool
    sessions_dir: Path
    current_session: Path


# ---------------------- Поддержка директорий ----------------------

def path_is_symlink(p: Path) -> bool:
    """
    Возвращает True для символических ссылок/джанкшенов; внутрь не входим. 
    """
    return p.is_symlink()


def windows_path_too_long(p: Path) -> bool:
    """
    Грубая проверка на превышение лимита длины пути Windows (≈260). 
    """
    if sys.platform != "win32":
        return False
    return len(str(p)) >= 248


def dir_max_mtime(dir_path: Path) -> float:
    """
    Возвращает максимальный st_mtime внутри дерева каталога (без followlinks). 
    """
    latest = dir_path.stat().st_mtime
    for root, dirs, files in os.walk(dir_path, followlinks=False):
        rp = Path(root)
        try:
            latest = max(latest, rp.stat().st_mtime)
        except Exception:
            pass
        for f in files:
            fp = rp / f
            try:
                latest = max(latest, fp.stat().st_mtime)
            except Exception:
                pass
    return latest


def dir_is_stable(dir_path: Path, quiet_sec: int) -> bool:
    """
    Считает каталог стабильным, если max mtime старше quiet_sec секунд. 
    """
    try:
        last = dir_max_mtime(dir_path)
        return (time.time() - last) >= quiet_sec
    except Exception:
        return False


def compress_directory_to_zip(src_dir: Path,
                              dst_zip: Path,
                              method: str,
                              level: int,
                              dry_run: bool) -> Optional[Path]:
    """
    Упаковывает каталог в ZIP с выбранным методом и уровнем; arcname — относительный путь. 
    """
    method_map = {
        "ZIP_STORED": zipfile.ZIP_STORED,
        "ZIP_DEFLATED": zipfile.ZIP_DEFLATED,
        "ZIP_BZIP2": zipfile.ZIP_BZIP2,
        "ZIP_LZMA": zipfile.ZIP_LZMA,
    }
    comp = method_map.get(method.upper(), zipfile.ZIP_DEFLATED)
    dst_zip = dst_zip.with_suffix(".zip")

    if dry_run:
        logging.info("[DRY] ZIP: %s -> %s (method=%s, level=%s)",
                     src_dir, dst_zip, method, level)
        return dst_zip

    dst_zip.parent.mkdir(parents=True, exist_ok=True)

    # compresslevel работает для DEFLATED/BZIP2; для STORED/LZMA игнорируется.
    kwargs = {}
    if comp in (zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2):
        kwargs["compresslevel"] = int(level)

    with zipfile.ZipFile(dst_zip, mode="w", compression=comp, **kwargs) as zf:
        for root, _, files in os.walk(src_dir, followlinks=False):
            rp = Path(root)
            for f in files:
                fp = rp / f
                try:
                    arcname = fp.relative_to(src_dir)
                except Exception:
                    arcname = fp.name
                zf.write(fp, arcname.as_posix())

    logging.info("ZIP created: %s", dst_zip)
    return dst_zip


def important_dir_by_rules(dir_path: Path, cfg: dict) -> bool:
    """
    Простая эвристика "важности" каталога: имя по high_keywords или "свежий" mtime. 
    """
    name = dir_path.name.lower()
    rules = cfg.get("importance_rules", {})
    high_kw = [k.lower() for k in rules.get("high_keywords", [])]
    if any(k in name for k in high_kw):
        return True
    days = int(rules.get("recent_days_for_high", 7))
    try:
        last = dir_max_mtime(dir_path)
        if (time.time() - last) <= days * 86400:
            return True
    except Exception:
        return False
    return False


def dated_root(root_dir: Path) -> Path:
    """
    Возвращает/создаёт Desktop_Cleanup_YYYY-MM-DD внутри root_cleanup_dir.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    dst = root_dir / f"Desktop_Cleanup_{today}"
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def safe_move_any(ctx: CleanerContext, src: Path, dst_dir: Path) -> Optional[Path]:
    """
    Перемещает файл/папку с уникализацией имени и логированием (или [DRY]).
    """
    dst = unique_destination(dst_dir, src.name)
    if ctx.dry_run:
        logging.info("[DRY] Move: %s -> %s", src, dst)
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    logging.info("Moved: %s -> %s", src, dst)
    return dst


# ---------------------- Обработка одного файла/папки ----------------------

def process_one(ctx: CleanerContext, p: Path) -> Optional[Dict]:
    """
    Обрабатывает один файл: исключения → 0‑байт → стабильность → категория/важность → перемещение.
    """
    if not p.exists() or not p.is_file():
        return None

    if matches_exclusions(p, ctx.cfg, ctx.cleanup_root):
        return None

    if is_zero_byte(p):
        if ctx.dry_run:
            logging.info("[DRY] Delete zero-byte: %s", p)
        else:
            try:
                p.unlink(missing_ok=True)
                logging.info("Deleted zero-byte: %s", p)
            except Exception as e:
                logging.warning("Failed to delete zero-byte %s: %s", p, e)
        return None

    if not is_stable(p, ctx.stable_attempts, ctx.stable_interval):
        logging.info("Skip unstable (retry on next event): %s", p)
        return None

    cat = categorize(p, ctx.cfg)
    important_rule = important_by_rules(p, ctx.cfg)
    important_ml = predict_importance_ml(p.name, ctx.model_path) if ctx.use_ml else None
    important = important_ml if important_ml is not None else important_rule

    root = dated_root(ctx.cleanup_root)
    bucket = "Important" if important else "Unimportant"
    target_dir = root / bucket / cat

    dst = safe_move_any(ctx, p, target_dir)
    if dst is None:
        return None

    return {
        "src": str(p),
        "dst": str(dst),
        "category": cat,
        "important": important,
        "time": datetime.now().isoformat(),
        "action": "move_file",
    }


def process_dir(ctx: CleanerContext, d: Path) -> Optional[Dict]:
    """
    Обрабатывает директорию как единое целое: стабилизация → ZIP или перемещение,
    без "разборки" содержимого, с защитой от симлинков и длинных путей. 
    """
    if not d.exists() or not d.is_dir():
        return None

    if matches_exclusions(d, ctx.cfg, ctx.cleanup_root):
        return None

    if path_is_symlink(d):
        logging.info("Skip symlink/junction: %s", d)
        return None

    folders_cfg = ctx.cfg.get("folders", {})
    if folders_cfg.get("handle", "ignore") != "atomic":
        return None

    # Удаление пустых папок (настраиваемо).
    if folders_cfg.get("delete_empty_folders", True):
        try:
            if not any(d.iterdir()):
                if ctx.dry_run:
                    logging.info("[DRY] Rmdir empty: %s", d)
                else:
                    d.rmdir()
                    logging.info("Rmdir empty: %s", d)
                return None
        except Exception:
            pass

    # "Тихий период" перед операцией: каталог должен быть стабилен.
    quiet = int(folders_cfg.get("quiet_period_sec", 30))
    if not dir_is_stable(d, quiet):
        logging.info("Skip unstable directory (quiet %ss not met): %s", quiet, d)
        return None

    # План назначения: Important/Unimportant + категория "folders".
    root = dated_root(ctx.cleanup_root)
    bucket = "Important" if important_dir_by_rules(d, ctx.cfg) else "Unimportant"
    folders_cat = folders_cfg.get("category_name", "folders")
    target_dir = root / bucket / folders_cat
    tentative_dst = unique_destination(target_dir, d.name)

    # Лимит длины путей Windows.
    if windows_path_too_long(tentative_dst):
        logging.warning("Skip directory (Windows MAX_PATH risk): %s", tentative_dst)
        return None

    # Сжатие или перемещение целиком.
    comp_cfg = folders_cfg.get("compress", {})
    if comp_cfg.get("enabled", False):
        method = comp_cfg.get("method", "ZIP_DEFLATED")
        level = int(comp_cfg.get("level", 6))
        dst_zip = tentative_dst.with_suffix(".zip")

        z = compress_directory_to_zip(d, dst_zip, method, level, ctx.dry_run)
        if z is None:
            return None

        if comp_cfg.get("remove_original_after_zip", False):
            if ctx.dry_run:
                logging.info("[DRY] Rmtree after ZIP: %s", d)
            else:
                try:
                    shutil.rmtree(d)
                    logging.info("Removed source directory after ZIP: %s", d)
                except Exception as e:
                    logging.warning("Failed to remove source directory %s: %s", d, e)

        return {
            "src": str(d),
            "dst": str(z),
            "category": folders_cat,
            "important": (bucket == "Important"),
            "time": datetime.now().isoformat(),
            "action": "zip_dir",
        }

    dst = safe_move_any(ctx, d, target_dir)
    if dst is None:
        return None

    return {
        "src": str(d),
        "dst": str(dst),
        "category": folders_cat,
        "important": (bucket == "Important"),
        "time": datetime.now().isoformat(),
        "action": "move_dir",
    }


# ---------------------- Watchdog: очередь директорий и обработчик ----------------------

class DirQueue:
    """
    Потокобезопасная очередь/карта директорий для отложенной обработки после
    "тихого периода" (N секунд без событий), чтобы избежать гонок. 
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, float] = {}

    def mark(self, path: Path) -> None:
        """
        Помечает директорию как "ожидающую стабилизации", сохраняя время события. 
        """
        with self._lock:
            self._pending[str(path)] = time.time()

    def due(self, quiet_sec: int) -> List[Path]:
        """
        Возвращает список директорий, чей "тихий период" истёк, удаляя их из
        очереди; вызывается периодически в главном цикле watch. 
        """
        now = time.time()
        due_list = []
        with self._lock:
            for s, t in list(self._pending.items()):
                if (now - t) >= quiet_sec:
                    due_list.append(Path(s))
                    del self._pending[s]
        return due_list


class DesktopHandler(FileSystemEventHandler):
    """
    Обработчик событий watchdog: on_created/on_moved для файлов и каталогов,
    on_modified у директорий — только для продления "тихого периода". 
    """

    def __init__(self, ctx: CleanerContext, dir_queue: DirQueue, quiet_sec: int):
        super().__init__()
        self.ctx = ctx
        self.dir_queue = dir_queue
        self.quiet_sec = quiet_sec

    def on_created(self, event):
        if event.is_directory:  # Каталоги откладываем до стабилизации. 
            self.dir_queue.mark(Path(event.src_path))
            return
        rec = process_one(self.ctx, Path(event.src_path))
        if rec:
            append_session_record(self.ctx.current_session, [rec])

    def on_moved(self, event):
        if event.is_directory:  # Новая позиция каталога — тоже в очередь. 
            self.dir_queue.mark(Path(event.dest_path))
            return
        rec = process_one(self.ctx, Path(event.dest_path))
        if rec:
            append_session_record(self.ctx.current_session, [rec])

    def on_modified(self, event):
        if event.is_directory:  # Любая модификация каталога — продлеваем "тишину".
            self.dir_queue.mark(Path(event.src_path))
        return


# ---------------------- CLI-команды ----------------------

def build_ctx(cfg: dict) -> CleanerContext:
    """
    Создаёт контекст обработки, настраивает логирование и файл журнала сессии.
    """
    desktop = expand(cfg.get("desktop_dir", "~/Desktop"))
    cleanup_root = expand(cfg.get("root_cleanup_dir", "~/Desktop_Cleanup"))
    quarantine = cleanup_root / "Quarantine"
    sessions_dir = expand(cfg.get("sessions_dir", "~/.desktop_butler/sessions"))
    log_dir = expand(cfg.get("log_dir", "~/.desktop_butler/logs"))

    setup_logging(log_dir)

    return CleanerContext(
        cfg=cfg,
        desktop=desktop,
        cleanup_root=cleanup_root,
        quarantine_dir=quarantine,
        dry_run=bool(cfg.get("dry_run", True)),
        stable_attempts=int(cfg.get("stable_check_attempts", 3)),
        stable_interval=float(cfg.get("stable_check_interval_sec", 1.0)),
        model_path=expand(cfg.get("model_path",
                                  "~/.desktop_butler/importance_model.joblib")),
        use_ml=bool(cfg.get("use_ml_importance", False)),
        sessions_dir=sessions_dir,
        current_session=session_file(sessions_dir),
    )


def cmd_scan_once(cfg: dict):
    """
    Разовая уборка: сначала директории (атомарно), затем файлы (по категориям).
    """
    ctx = build_ctx(cfg)
    sess = ctx.current_session
    records = []

    # Важно сначала обрабатывать каталоги, чтобы избежать частичной "разборки".
    for item in Path(ctx.desktop).iterdir():
        if item.is_dir():
            rec = process_dir(ctx, item)
            if rec:
                records.append(rec)

    for item in Path(ctx.desktop).iterdir():
        if item.is_file():
            rec = process_one(ctx, item)
            if rec:
                records.append(rec)

    if records:
        append_session_record(sess, records)
    logging.info("Scan once done. Records: %d", len(records))


def cmd_watch(cfg: dict):
    """
    Наблюдение в реальном времени: обрабатывает файлы сразу, директории —
    после "тихого периода" через очередь DirQueue.
    """
    ctx = build_ctx(cfg)
    folders_cfg = ctx.cfg.get("folders", {})
    quiet_dirs = int(folders_cfg.get("quiet_period_sec", 30))

    dq = DirQueue()
    observer = Observer()
    handler = DesktopHandler(ctx, dq, quiet_dirs)
    observer.schedule(handler, str(ctx.desktop), recursive=False)
    observer.start()

    logging.info("Watching %s ... Press Ctrl+C to stop.", ctx.desktop)

    try:
        while True:
            # Каталоги, у которых истёк "тихий период", обрабатываем атомарно.
            for d in dq.due(quiet_dirs):
                rec = process_dir(ctx, d)
                if rec:
                    append_session_record(ctx.current_session, [rec])
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
        observer.stop()
    observer.join()


def cmd_train_importance(cfg: dict, labeled_csv: Path):
    """
    Обучение модели важности на CSV (path,label) и сохранение через joblib.
    """
    model_path = expand(cfg["model_path"])
    train_importance_model(labeled_csv, model_path)
    logging.info("Trained and saved importance model to %s", model_path)


def cmd_predict_check(cfg: dict, name: str):
    """
    Диагностика: предсказание важности для имени файла по сохранённой модели.
    """
    model_path = expand(cfg["model_path"])
    pred = predict_importance_ml(name, model_path)
    logging.info("Predict importance for '%s': %s", name, pred)


def cmd_undo_last(cfg: dict):
    """
    Откат последней сессии перемещений по журналу session_*.json (LIFO).
    """
    sessions_dir = expand(cfg["sessions_dir"])
    last = get_last_session(sessions_dir)
    if not last:
        logging.info("No session to undo.")
        return
    data = json.loads(last.read_text(encoding="utf-8"))
    undone = 0
    for rec in reversed(data):
        src = Path(rec["dst"])
        dst = Path(rec["src"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
                undone += 1
            except Exception as e:
                logging.warning("Undo failed %s -> %s: %s", src, dst, e)
    logging.info("Undo complete. Restored: %d", undone)


def cmd_purge_quarantine(cfg: dict):
    """
    Очистка карантина: удалить файлы старше quarantine_days; возвращает счётчик.
    """
    quarantine_dir = expand(cfg.get("root_cleanup_dir",
                                    "~/Desktop_Cleanup")) / "Quarantine"
    days = int(cfg.get("quarantine_days", 14))
    count = purge_quarantine(quarantine_dir, days)
    logging.info("Quarantine purge: removed %d files older than %d days.",
                 count, days)


def main():
    """
    CLI: глобальная опция --config указывается перед подкомандой (argparse).
    """
    parser = argparse.ArgumentParser(prog="Desktop Butler")
    parser.add_argument("--config", required=True, help="Path to config.yaml")

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan-once")
    sub.add_parser("watch")

    ptrain = sub.add_parser("train-importance")
    ptrain.add_argument("--labeled", required=True, help="CSV with path,label")

    pcheck = sub.add_parser("predict-check")
    pcheck.add_argument("--name", required=True, help="Filename to check")

    sub.add_parser("undo-last")
    sub.add_parser("purge-quarantine")

    args = parser.parse_args()
    cfg = load_config(expand(args.config))

    if args.cmd == "scan-once":
        cmd_scan_once(cfg)
    elif args.cmd == "watch":
        cmd_watch(cfg)
    elif args.cmd == "train-importance":
        cmd_train_importance(cfg, Path(args.labeled))
    elif args.cmd == "predict-check":
        cmd_predict_check(cfg, args.name)
    elif args.cmd == "undo-last":
        cmd_undo_last(cfg)
    elif args.cmd == "purge-quarantine":
        cmd_purge_quarantine(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
