"""
Setup script for TextureSynth wheel packaging.
Packages the pre-built PyBind11 C++ Vulkan module from CMake.

Version is read from ADDON/blender_manifest.toml.
"""

import os
import sys
import re
import glob
import shutil
from setuptools import setup, Distribution
from setuptools.command.build_ext import build_ext

def _get_version() -> str:
    v = os.environ.get('TEXTURESYNTH_VERSION')
    if not v:
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'ADDON', 'blender_manifest.toml')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                for line in f:
                    m = re.match(r'^version\s*=\s*"([^"]+)"', line)
                    if m:
                        v = m.group(1)
                        break
        if not v:
            v = '0.0.0'
    return v

VERSION = _get_version()
print(f"[TextureSynth] Building version {VERSION}")

class BinaryDistribution(Distribution):
    """Force a binary (platform-tagged) wheel."""
    def has_ext_modules(self):
        return True

class CopyPrebuiltExt(build_ext):
    """Copy pre-built extension instead of compiling."""

    def run(self):
        ext_suffix = '.pyd' if sys.platform == 'win32' else '.so'
        base_dir = os.path.dirname(os.path.abspath(__file__))

        module_glob = f'texturesynth_core*{ext_suffix}'
        patterns = [
            os.path.join(base_dir, 'build', 'Release', module_glob),
            os.path.join(base_dir, 'build', '**', 'Release', module_glob),
            os.path.join(base_dir, 'build', '**', module_glob),
        ]

        all_matches = []
        for pattern in patterns:
            all_matches.extend(glob.glob(pattern, recursive=True))
        all_matches = sorted(set(all_matches))

        exact_name = f'texturesynth_core{ext_suffix}'
        ext_file = None
        for m in all_matches:
            if os.path.basename(m) == exact_name:
                ext_file = m
                break
        if ext_file is None and all_matches:
            ext_file = all_matches[0]

        if not ext_file:
            print("ERROR: Could not find pre-built texturesynth_core extension!")
            sys.exit(1)

        self.build_lib = self.build_lib or 'build/lib'
        os.makedirs(self.build_lib, exist_ok=True)

        dest = os.path.join(self.build_lib, exact_name)
        print(f"Copying {ext_file} -> {dest}")
        shutil.copy2(ext_file, dest)

setup(
    name='texturesynth',
    version=VERSION,
    author='User',
    description='Node-based Vulkan Procedural Texture Generator',
    packages=[],
    cmdclass={'build_ext': CopyPrebuiltExt},
    distclass=BinaryDistribution,
    python_requires='>=3.11',
    zip_safe=False,
)
