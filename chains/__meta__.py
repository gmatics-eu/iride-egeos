# `name` is the name of the package as used for `pip install package`
name = "chains"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "1.0.dev0"
author = "GMATICS srl"
author_email = "services@gmatics.eu"
description = ""  # One-liner
url = ""  # your project homepage
license = "MIT"  # See https://choosealicense.com
