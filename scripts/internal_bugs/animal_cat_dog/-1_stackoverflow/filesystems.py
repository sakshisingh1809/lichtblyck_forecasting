# Requires Python 3.6+

import os
import re


class FileSystem(object):
    class NoAccess(Exception):
        pass

    class Unknown(Exception):
        pass

    # Pattern for matching "xxx://"  # x is any non-whitespace character except for ":".
    _PATH_PREFIX_PATTERN = re.compile(r"\s*([^:]+)://")
    _registry = {}  # Registered subclasses.

    @classmethod
    def __init_subclass__(cls, /, path_prefix, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[path_prefix] = cls  # Add class to registry.

    @classmethod
    def _get_prefix(cls, s):
        """Extract any file system prefix at beginning of string s and
        return a lowercase version of it or None when there isn't one.
        """
        match = cls._PATH_PREFIX_PATTERN.match(s)
        return match.group(1).lower() if match else None

    def __new__(cls, path):
        """Create instance of appropriate subclass."""
        path_prefix = cls._get_prefix(path)
        subclass = cls._registry.get(path_prefix)
        if subclass:
            return object.__new__(subclass)
        else:  # No subclass with matching prefix found (and no default).
            raise cls.Unknown(f'path "{path}" has no known file system prefix')

    def count_files(self):
        raise NotImplementedError


class Nfs(FileSystem, path_prefix="nfs"):
    def __init__(self, path):
        pass

    def count_files(self):
        pass


class Ufs(Nfs, path_prefix="ufs"):
    def __init__(self, path):
        pass

    def count_files(self):
        pass


class LocalDrive(FileSystem, path_prefix=None):  # Default file system.
    def __init__(self, path):
        if not os.access(path, os.R_OK):
            raise self.NoAccess(f"Cannot read directory {path!r}")
        self.path = path

    def count_files(self):
        return sum(
            os.path.isfile(os.path.join(self.path, filename))
            for filename in os.listdir(self.path)
        )


if __name__ == "__main__":

    data1 = FileSystem("nfs://192.168.1.18")
    data2 = FileSystem("c:/")  # Change as necessary for testing.
    data4 = FileSystem("ufs://192.168.1.18")

    print(type(data1))  # -> <class '__main__.Nfs'>
    print(type(data2))  # -> <class '__main__.LocalDrive'>
    print(f"file count: {data2.count_files()}")  # -> file count: <some number>

    try:
        data3 = FileSystem("c:/foobar")  # A non-existent directory.
    except FileSystem.NoAccess as exc:
        print(f"{exc} - FileSystem.NoAccess exception raised as expected")
    else:
        raise RuntimeError("Non-existent path should have raised Exception!")

    try:
        data4 = FileSystem("foobar://42")  # Unregistered path prefix.
    except FileSystem.Unknown as exc:
        print(f"{exc} - FileSystem.Unknown exception raised as expected")
    else:
        raise RuntimeError("Unregistered path prefix should have raised Exception!")
