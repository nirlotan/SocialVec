#!/usr/bin/env python

"""The setup script."""

from setuptools import setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

# Keep requirements.txt in sync with these.
requirements = ['click>=8.1.3',
                'pandas>=1.5.0',
                'numpy>=1.23.3',
                'fastparquet>=0.8.3',
                'pyarrow',
                'requests',
                'yaspin',
                'PyYAML',
                'gensim>4',
                'scikit-learn']

extras_requirements = {
    # SocialVecClassifier
    'classifier': ['tensorflow'],
    # socialvec.api (the Flask server)
    'api': ['flask', 'flask-cors'],
}


setup(
    author="Nir Lotan",
    author_email='nir.lotan@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Sociology',
    ],
    description="Pre-trained social embeddings of popular Twitter accounts, for eliciting social world knowledge.",
    long_description=readme,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'socialvec=socialvec.cli:run',
        ],
    },
    install_requires=requirements,
    extras_require=extras_requirements,
    license="MIT",
    license_files=["LICENSE"],
    include_package_data=True,
    keywords=['socialvec', 'embeddings', 'social network', 'twitter', 'knowledge', 'word2vec'],
    name='socialvec',
    # socialvec.models only holds the classifier .h5 files.
    packages=['socialvec', 'socialvec.models'],
    # The embedding models and metadata are downloaded on first use, not shipped in the package.
    package_data={'socialvec': ['config.yaml'], 'socialvec.models': ['*.h5']},
    url='https://github.com/nirlotan/SocialVec',
    project_urls={
        'Paper': 'https://doi.org/10.1371/journal.pone.0283700',
        'Source': 'https://github.com/nirlotan/SocialVec',
        'Issues': 'https://github.com/nirlotan/SocialVec/issues',
        'Changelog': 'https://github.com/nirlotan/SocialVec/blob/master/HISTORY.rst',
    },
    version='0.2.0',
    zip_safe=False,
)
