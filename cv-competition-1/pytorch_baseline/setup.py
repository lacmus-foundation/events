import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'compute_overlap',
        ['compute_overlap.pyx']
    ),
]


setuptools.setup(
    name             = 'lacmus-metrics',
    version          = '1.0.0',
    description      = 'Lacmus object detection metrics.',
    url              = 'https://github.com/lacmus-foundation',
    author           = 'Georgy Perevozchikov',
    author_email     = 'gosha20777@live.ru',
    maintainer       = 'Georgy Perevozchikov',
    maintainer_email = 'gosha20777@live.ru',
    cmdclass         = {'build_ext': BuildExtension},
    packages         = setuptools.find_packages(),
    install_requires = ['cython'],
    entry_points     = {},
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)