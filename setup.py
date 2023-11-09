#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['click>=8.1.3',
                'setuptools>=60.2.0',
                'pandas>=1.5.0',
                'numpy>=1.23.3',
                'fastparquet>=0.8.3',
                'pyarrow',
                'wget>=3.2', 
                'yaspin', 
                'PyYAML', 
                'gensim>4', 
                'scikit-learn',
                'tensorflow']

test_requirements = [ ]

#excluded = [ 'socialvec/*.gz' ]

setup(
    author="Nir Lotan",
    author_email='nir.lotan@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="SocialVec is a framework of Social Embeddings for eliciting social world knowledge from social networks.",
    entry_points={
        'console_scripts': [
            'socialvec=socialvec.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='socialvec',
    name='socialvec',
    packages=find_packages(include=['socialvec', 'socialvec.*']),
    package_data={'your_package': ['*.gz', '*.h5']},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nirlotan/socialvec',
    version='0.1.6',
    zip_safe=False,
)
