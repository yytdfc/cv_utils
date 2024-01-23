from setuptools import setup, find_packages

PKG_NAME = 'fastcv'

fastcv = __import__(PKG_NAME)

requires = [
    "Pillow>=10.0.0",
    "opencv-python",
    "torch",
    "torchvision",
    "einops",
    "ffmpeg-python",
    "sh",
    "face-alignment",
]

setup(
    name=PKG_NAME,
    version=fastcv.__version__,
    description=fastcv.__description__,
    long_description=open('./README.md').read(),
    maintainer='yytdfc',
    maintainer_email='fuchen@foxmail.com',
    keywords=[],
    url='https://github.com/yytdfc/fastcv',
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    install_requires=requires,
    setup_requires=requires,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
