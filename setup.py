from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='jetgraphs',
    url='https://github.com/alessiodevoto/jetgraphs.git',
    author='Alessio Devoto',
    # Needed to actually package something
    packages=['jetgraphs'],
    # Needed for dependencies
    install_requires=[
        'numpy',
        'matplotlib==3.2.2',
        'networkx==2.6.3',
        'pytorch_lightning==1.7.1',
        'scikit-learn==1.0.2',
        'scipy==1.7.3',
        'torch-geometric==2.0.4',
        'torchmetrics==0.9.3',
        'tqdm'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    # license='MIT',
    description='A package to manipulate and plot jetgraphs',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.txt').read(),
)