import os
from enums.ConfigurationMode import ConfigurationMode

_conf = os.environ.get("CONFIGURATION", None)
if _conf is not None:
    CONFIGURATION = ConfigurationMode.DEBUG
else:
    CONFIGURATION = ConfigurationMode.RELEASE
# endregion

DEBUG = CONFIGURATION is ConfigurationMode.DEBUG
LOG_SHOW_INSPECT = CONFIGURATION is ConfigurationMode.DEBUG
