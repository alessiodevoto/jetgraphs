from setuptools import setup

setup(
    name='jetgraphs',
    url='https://github.com/alessiodevoto/jetgraphs.git',
    author='Alessio Devoto',
    packages=['jetgraphs'],
    install_requires=[
        'numpy',
        'matplotlib>=3.2.2',
        'networkx',
        'pytorch_lightning',
        'scikit-learn',
        'scipy',
        'torchmetrics',
        'tqdm'], # TODO add here torch geometric!
    version='0.1',
    # The license can be anything you like
    # license='MIT',
    description='A package to manipulate and plot jetgraphs',
    long_description=open('README.txt').read(),
)