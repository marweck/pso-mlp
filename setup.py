from setuptools import setup

setup(
    name='pso-nn',
    version='',
    packages=['pso'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Marcio Carvalho',
    author_email='marweck@gmail.com',
    description='',
    check_format=True,
    # Enable type checking
    test_mypy=False,
    # Enable linting at build time
    test_flake8=True,
)
