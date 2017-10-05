from setuptools import setup

setup(name='opthelp',
        version='0.0.1',
        url='https://github.com/lwcook/opt-help',
        download_url='https://github.com/lwcook/opt-help/archive/0.0.1.tar.gz',
        author='Laurence W. Cook',
        author_email='lwc24@cam.ac.uk',
        install_requires=['numpy >= 1.12.1'],
        license='MIT',
        packages=['opthelp'],
        include_package_data=True,
        zip_safe=False)
