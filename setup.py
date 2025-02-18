from setuptools import setup

setup(
    name='jetgraphs',
    url='https://github.com/alessiodevoto/jetgraphs.git',
    author='Alessio Devoto',
    packages=['jetgraphs'],
    install_requires=[
        'numpy==1.26.4',
        'matplotlib==3.5.1',
        'networkx',
        'pytorch_lightning',
        'scikit-learn',
        'scipy',
        'seaborn',
        'deepdish',
        'torchmetrics',
        'tqdm'], # TODO add here torch geometric!
    version='0.1',
    license='MIT',
    description='A package to manipulate and plot jetgraphs',
    long_description=open('README.md').read(),
)