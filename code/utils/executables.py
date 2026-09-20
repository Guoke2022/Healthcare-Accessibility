# -*- coding: utf-8 -*-
"""Module utilities for executables."""


from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            key = str(p.expanduser().resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _candidate_filenames(name: str) -> list[str]:
    p = Path(name)
    base = p.name
    if os.name == "nt" and not base.lower().endswith(".exe"):
        return [base, base + ".exe"]
    return [base]


def _prefix_bin_candidates(prefix: Path, filenames: list[str]) -> list[Path]:
    dirs = [
        prefix / "Library" / "bin",  # Conda/Windows
        prefix / "Scripts",          # Conda/Windows
        prefix / "bin",              # Unix / some Windows packages
        prefix,                       # explicit portable layout
    ]
    return [d / fn for d in dirs for fn in filenames]


def _infer_conda_prefixes(search_all_envs: bool = True) -> list[Path]:
    """Helper for _infer_conda_prefixes."""
    prefixes: list[Path] = [Path(sys.prefix)]

    for key, value in os.environ.items():
        if key == "CONDA_PREFIX" or key.startswith("CONDA_PREFIX_"):
            if value:
                prefixes.append(Path(value))


    roots: list[Path] = []
    for p in list(prefixes):
        if p.parent.name.lower() == "envs":
            roots.append(p.parent.parent)
        else:
            roots.append(p)


    home = Path.home()
    roots.extend([
        home / "miniforge3",
        home / "mambaforge",
        home / "miniconda3",
        home / "anaconda3",
    ])
    if os.name == "nt":
        roots.extend([
            Path(r"C:\ProgramData\miniforge3"),
            Path(r"C:\ProgramData\mambaforge"),
            Path(r"C:\ProgramData\miniconda3"),
            Path(r"C:\ProgramData\anaconda3"),
        ])

    roots = _dedupe_paths(roots)
    prefixes.extend(roots)

    if search_all_envs:
        for root in roots:
            envs_dir = root / "envs"
            if not envs_dir.is_dir():
                continue
            try:
                prefixes.extend(p for p in envs_dir.iterdir() if p.is_dir())
            except OSError:
                pass

    return _dedupe_paths(prefixes)


def resolve_executable(
    name: str,
    *,
    env_var: str | None = None,
    project_root: Path | None = None,
    extra_candidates: Iterable[Path] = (),
    search_all_conda_envs: bool = True,
    install_hint: str | None = None,
) -> str:
    """Helper for resolve_executable."""


    attempted: list[str] = []
    filenames = _candidate_filenames(name)

    def _accept_path(p: Path) -> str | None:
        p = p.expanduser()
        attempted.append(str(p))
        if p.is_file():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
        return None


    if env_var:
        raw = os.environ.get(env_var, "").strip().strip('"')
        if raw:
            p = Path(raw)
            hit = _accept_path(p)
            if hit:
                return hit
            found = shutil.which(raw) or (shutil.which(raw + ".exe") if os.name == "nt" else None)
            if found:
                return found
            attempted.append(f"{env_var}={raw} (not found)")


    if any(sep in name for sep in ("/", "\\")) or Path(name).suffix:
        hit = _accept_path(Path(name))
        if hit:
            return hit


    if project_root is not None:
        pr = Path(project_root)
        project_candidates: list[Path] = []
        for fn in filenames:
            project_candidates.extend([
                pr / "tools" / Path(name).stem / fn,
                pr / "tools" / "bin" / fn,
                pr / "tools" / fn,
            ])
        for c in project_candidates:
            hit = _accept_path(c)
            if hit:
                return hit

    # 4. PATH。
    for cmd in filenames:
        found = shutil.which(cmd)
        attempted.append(f"PATH:{cmd}")
        if found:
            return found


    for prefix in _infer_conda_prefixes(search_all_envs=search_all_conda_envs):
        for c in _prefix_bin_candidates(prefix, filenames):
            hit = _accept_path(c)
            if hit:
                return hit


    for c in extra_candidates:
        hit = _accept_path(Path(c))
        if hit:
            return hit

    env_text = f"环境变量 {env_var}" if env_var else "对应环境变量"
    tail = "\n".join(f"  - {x}" for x in attempted[-20:])
    hint = f"\n{install_hint}" if install_hint else ""
    raise FileNotFoundError(
        f"找不到可执行程序：{name}\n"
        f"已经检查 PATH、当前 Python/Conda 环境、Conda base/sibling envs 和项目 tools 目录。\n"
        f"你也可以通过 {env_text} 指向完整可执行文件。{hint}\n"
        f"最后检查的候选位置：\n{tail}"
    )


def resolve_osmium(project_root: Path) -> str:
    exe = resolve_executable(
        "osmium",
        env_var="NC_OSMIUM_EXE",
        project_root=project_root,
        install_hint=(
            "若当前环境确实未安装，可在已激活的 NC_appeal 环境执行：\n"
            "  conda install -c conda-forge osmium-tool\n"
            "若 osmium 已安装在其他位置，可设置 NC_OSMIUM_EXE 指向原安装目录中的 osmium.exe。"
        ),
    )


    if os.name == "nt":
        p = Path(exe)
        if p.name.lower() == "osmium.exe":
            exe = str(p.with_name("osmium.exe"))

    return exe


def resolve_cargo(project_root: Path) -> str:
    return resolve_executable(
        "cargo",
        env_var="NC_CARGO_EXE",
        project_root=project_root,
        extra_candidates=[
            Path.home() / ".cargo" / "bin" / "cargo.exe",
            Path.home() / ".cargo" / "bin" / "cargo",
        ],
        install_hint=(
            "若未安装 Rust，请安装 rustup；若已安装但不在 PATH，可设置 NC_CARGO_EXE。"
        ),
    )


def executable_version(exe: str) -> str:
    """Helper for executable_version."""
    p = subprocess.run(
        [exe, "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = (p.stdout or "").strip()
    if p.returncode != 0:
        raise RuntimeError(
            f"找到可执行程序但无法正常启动：{exe}\n"
            f"returncode={p.returncode}\n{text}"
        )
    return text.splitlines()[0] if text else "version unknown"
