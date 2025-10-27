from setuptools import find_packages
from setuptools import setup


with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(name='digital-shield-packages',
      version="0.1.0",
      description="ML-based cybersecurity project with financial loss and severity prediction models",
      license="MIT",
      author="Digital Shield Team",
      packages=find_packages(),
      install_requires=requirements,
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
