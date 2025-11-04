#!/usr/bin/env python
from pathlib import Path
from typing import List
from setuptools import setup
import versioneer


def read_requirements(path: str) -> List[str]:
    """Return non-empty, comment-stripped requirements from *path*."""
    requirements = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


if __name__ == "__main__":
    setup(
        version="0.6.4",
        install_requires=read_requirements("requirements.txt"),
    )

