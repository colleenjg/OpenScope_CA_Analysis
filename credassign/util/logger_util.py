#!/usr/bin/env python

"""
logger_util.py

This module contains logging functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.

"""

import copy
import logging
import os
import sys
from pathlib import Path
import warnings


ORIGINAL_WARNINGS_FORMAT = copy.deepcopy(warnings.formatwarning)


#############################################
class StoreRootLoggingInfo():
    """
    Context manager for temporarily storing root logging information in global 
    variables, along with warnings format information. 
    
    This is useful if joblib's Parallel() is called with the loky backend, as 
    logger handlers and level are reset within the parallel processes.

    Optional init args:
        - warn_config (bool): if True, a warning is sent if the root logger 
                              configuration or the warnings format is not 
                              recognized.
                              default: True
    """

    def __init__(self, warn_config=True, extra_warn_msg=""):

        self.warn_config = warn_config
        self.extra_warn_msg = extra_warn_msg
    

    def __enter__(self):

        # get the root logger
        logger = logging.getLogger()

        # store a general variable
        self.remove_gen_var = True
        if "SET_ROOT_LOGGING" in os.environ.keys():
            self.remove_gen_var = False
            self.prev_root_gen = os.environ.get("SET_ROOT_LOGGING")
        os.environ["SET_ROOT_LOGGING"] = "1"

        # store config
        self.remove_root_config_var = True
        if "SET_ROOT_LOGGING_CONFIG" in os.environ.keys():
            self.remove_root_config_var = False
            self.prev_root_config = os.environ.get("SET_ROOT_LOGGING_CONFIG")

        if (len(logger.handlers) == 1 and 
            isinstance(logger.handlers[0], logging.StreamHandler) and 
            isinstance(logger.handlers[0].formatter, BasicLogFormatter)):
            os.environ["SET_ROOT_LOGGING_CONFIG"] = "basic"
        elif self.warn_config:
            warnings.warn(
                f"Logging configuration not identified.{self.extra_warn_msg}"
                )

        # store logging level
        log_level = logger.level

        self.remove_root_level_var = True
        if "SET_ROOT_LOGGING_LEVEL" in os.environ.keys():
            self.remove_root_level_var = False
            self.prev_root_level = os.environ.get("SET_ROOT_LOGGING_LEVEL")
        os.environ["SET_ROOT_LOGGING_LEVEL"] = str(log_level)

        # store warnings format
        self.remove_warnings_fmt = True
        if "SET_WARNINGS_FORMAT" in os.environ.keys():
            self.remove_warnings_fmt = False
            self.prev_warnings_fmt = os.environ.get("SET_WARNINGS_FORMAT")

        # not the best way to compare things, but should do the trick typically
        if warnings.formatwarning == warnings_simple:
            os.environ["SET_WARNINGS_FORMAT"] = "simple"
        elif (warnings.formatwarning != ORIGINAL_WARNINGS_FORMAT and 
            self.warn_config):
            warnings.warn(
                f"Warning formatting not identified.{self.extra_warn_msg}"
                )
        
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if "SET_ROOT_LOGGING" in os.environ.keys():
            if self.remove_gen_var:
                del os.environ["SET_ROOT_LOGGING"]
            else:
                os.environ["SET_ROOT_LOGGING"] = self.prev_root_gen

        if "SET_ROOT_LOGGING_CONFIG" in os.environ.keys():
            if self.remove_root_config_var:
                del os.environ["SET_ROOT_LOGGING_CONFIG"]
            else:
                os.environ["SET_ROOT_LOGGING_CONFIG"] = self.prev_root_config

        if "SET_WARNINGS_FORMAT" in os.environ.keys():
            if self.remove_warnings_fmt:
                del os.environ["SET_WARNINGS_FORMAT"]
            else:
                os.environ["SET_WARNINGS_FORMAT"] = self.prev_warnings_fmt

        if "SET_ROOT_LOGGING_LEVEL" in os.environ.keys():
            if self.remove_root_level_var:
                del os.environ["SET_ROOT_LOGGING_LEVEL"]
            else:
                os.environ["SET_ROOT_LOGGING_LEVEL"] = self.prev_root_level



#############################################
class TempWarningFilter():
    """
    Context manager for temporarily filtering specific warnings.

    Optional init args:
        - msgs (list)  : Beginning of message in the warning to filter. 
                         Must be the same length as categs.
                         default: []
        - categs (list): Categories of the warning to filter. Must be 
                         the same length as msgs.
                         default: []    
    """

    def __init__(self, msgs=[], categs=[]):
        self.orig_warnings = warnings.filters

        if not isinstance(msgs, list):
            msgs = [msgs]
        self.msgs = msgs


        if not isinstance(categs, list):
            categs = [categs]
        self.categs = categs

        if len(self.msgs) != len(self.categs):
            raise ValueError("Must provide as many 'msgs' as 'categs'.")


    def __enter__(self):
        for msg, categ in zip(self.msgs, self.categs):
            warnings.filterwarnings("ignore", message=msg, category=categ)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        warnings.filters = self.orig_warnings


#############################################
class TempChangeLogLevel():
    """
    Context manager for temporarily changing logging level.

    Optional init args:
        - logger (logger) : logging Logger object. If None, root logger is used.
                            default: None
        - level (int, str): logging level to temporarily set logger to.
                            If None,log level is not changed.
                            default: "info"
    """

    def __init__(self, logger=None, level="info"):

        if logger is None or logger.level == logging.NOTSET:
            logger = logging.getLogger()
        
        self.logger = logger
        self.level = level


    def __enter__(self):

        if self.level is not None:
            self.prev_level = self.logger.level
            set_level(level=self.level, logger=self.logger)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.level is not None:
            set_level(level=self.prev_level, logger=self.logger)


#############################################
def get_logger(logtype="stream", name=None, filename="logs.log", 
               fulldir=".", level="info", fmt=None, skip_exists=True):
    """
    get_logger()

    Returns logger with specified formatting. If no logger is passed, sets the 
    root logger.

    Optional args:
        - logtype (str)     : type or types of handlers to add to logger 
                              ("stream", "file", "both", "none")
                              default: "stream"
        - name (str)        : logger name. If None, the root logger is returned.
                              default: None
        - filename (str)    : name under which to save file handler, if it is 
                              included
                              default: "logs.log"
        - fulldir (str)     : path under which to save file handler, if it is
                              included
                              default: "."
        - level (str)       : level of the logger ("info", "error", "warning", 
                               "debug", "critical")
                              default: "info"
        - fmt (Formatter)   : logging Formatter to use for the handlers
                              default: None
        - skip_exists (bool): if a logger with the name already has the 
                              specified handlers, formats them, and returns 
                              existing logger. 
                              Otherwise, resets them.
                              default: True

    Returns:
        - logger (Logger): logger object
    """

    # create one instance
    if isinstance(name, logging.Logger):
        logger = name
    else:
        logger = logging.getLogger(name)
    
    if not skip_exists:
        logger.handlers = []

    # create handlers
    if logtype in ["stream", "both"]:
        add_handler = True
        for hd in logger.handlers:
            if isinstance(hd, logging.StreamHandler):
                add_handler = False
                if fmt is not None:
                    hd.setFormatter(fmt)
        if add_handler:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            logger.addHandler(sh)

    if logtype in ["file", "both"]:
        add_handler = True
        for hd in logger.handlers:
            if isinstance(hd, logging.FileHandler):
                add_handler = False
                if fmt is not None:
                    hd.setFormatter(fmt)
        if add_handler:
            fh = logging.FileHandler(Path(fulldir, filename))
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        
    all_types = ["file", "stream", "both", "none"]
    if logtype not in all_types:
        val_str = ", ".join([f"'{x}'" for x in all_types])        
        raise ValueError(
            f"'logtype' value '{logtype}' unsupported. Must be in {val_str}."
        )

    set_level(level, logger)
    
    return logger


#############################################
def get_logger_with_basic_format(**logger_kw):
    """
    get_logger_with_basic_format()

    Returns logger with basic formatting, defined by BasicLogFormatter class. 
    If no logger is passed, sets the root logger.

    Keyword args:
        - logger_kw (dict): keyword arguments for get_logger()
        
    Returns:
        - logger (Logger): logger object
    """


    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger
    
    
#############################################
def warnings_simple(message, category, filename, lineno, file=None, line=None):
    """
    warnings_simple(message, category, filename, lineno()

    Warning format that doesn't cite the line of code.
    Adapted from: https://pymotw.com/2/warnings/

    Required args: warnings module arguments
        
    Returns:
        - (str): formatting string
    """

    return '%s:%s: %s:\n%s\n' % (filename, lineno, category.__name__, message)


#############################################
#############################################
class BasicLogFormatter(logging.Formatter):
    """
    BasicLogFormatter()

    Basic formatting class that formats different level logs differently. 
    Allows a spacing extra argument to add space at the beginning of the log.
    """

    dbg_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(spacing)s%(msg)s"
    wrn_fmt  = "%(spacing)s%(levelname)s: %(msg)s"
    err_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    crt_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(spacing)s%(levelname)s: %(msg)s"):
        """
        Optional args:
            - fmt (str): default format style.
        """
        super().__init__(fmt=fmt, datefmt=None, style="%") 

    def format(self, record):

        if not hasattr(record, "spacing"):
            record.spacing = ""

        # Original format as default
        format_orig = self._style._fmt

        # Replace default as needed
        if record.levelno == logging.DEBUG:
            self._style._fmt = BasicLogFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = BasicLogFormatter.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = BasicLogFormatter.wrn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = BasicLogFormatter.err_fmt
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = BasicLogFormatter.crt_fmt

        # Call the original formatter class to do the grunt work
        formatted_log = logging.Formatter.format(self, record)

        # Restore default format
        self._style._fmt = format_orig

        return formatted_log


#############################################
def set_level(level="info", logger=None, return_only=False):
    """
    set_level()

    Sets level of the named logger, or of the root logger, otherwise.

    Optional args:
        - level (int or str): level of the logger ("info", "error", "warning", 
                              "debug", "critical", 10, 50)
                              default: "info"
        - logger (Logger)   : logging Logger. If None, the root logger is set.
                              default: None
        - return_only (bool): if True, level is not set, but only returned
                              default: False

    Returns:
        - level (int): logging level requested
    """
    
    if logger is None:
        # get the root logger
        logger = logging.getLogger()

    if isinstance(level, int):
        level = level
    elif isinstance(level, str) and level.isdigit():
        level = int(level)
    elif level.lower() == "debug":
        level = logging.DEBUG
    elif level.lower() == "info":
        level = logging.INFO
    elif level.lower() == "warning":
        level = logging.WARNING
    elif level.lower() == "error":
        level = logging.ERROR
    elif level.lower() == "critical":
        level = logging.CRITICAL
    else:
        accepted_values = ["debug", "info", "warning", "error", "critical"]
        val_str = ", ".join([f"'{x}'" for x in accepted_values])        
        raise ValueError(
            f"'level' value '{level}' unsupported. Must be in {val_str}."
        )

    if not return_only:
        logger.setLevel(level)

    return level


#############################################
def format_all(**logger_kw):
    """
    format_all()

    Initializes a logger with the basic format, and updates the warnings 
    format.
    """
    
    _ = get_logger_with_basic_format(**logger_kw)
    warnings.formatwarning = warnings_simple


#############################################
def get_module_logger(name=None):
    """
    get_module_logger()

    Initializes a module logger. Also, checks whether the root logger needs to 
    be reconfigurated, and have its level updated, based on global variables 
    that can be set with the context manager StoreRootLoggingInfo().

    Optional args:
        - name (str): module logger name
                      default: None
    """

    logger = logging.getLogger(name)

    if os.environ.get("SET_ROOT_LOGGING", "0") == "1":

        if "SET_ROOT_LOGGING_CONFIG" in os.environ.keys():
            config = os.environ["SET_ROOT_LOGGING_CONFIG"]
            if config == "basic":
                get_logger_with_basic_format()
            else:
                raise NotImplementedError(
                    f"{config} logging config not recognized."
                    )

        if "SET_ROOT_LOGGING_LEVEL" in os.environ.keys():
            level = os.environ["SET_ROOT_LOGGING_LEVEL"]
            set_level(level)

        if "SET_WARNINGS_FORMAT" in os.environ.keys():
            warn_format = os.environ["SET_WARNINGS_FORMAT"]
            if warn_format == "simple":
                warnings.formatwarning = warnings_simple
            else:
                raise NotImplementedError(
                    f"{warn_format} warnings format not recognized."
                    )

        os.environ["SET_ROOT_LOGGING"] = "0"

    return logger


#############################################
if __name__ == '__main__':

    format_all()
