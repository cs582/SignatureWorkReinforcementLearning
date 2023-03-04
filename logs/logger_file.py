import logging

# set up the loggers
logging.basicConfig(format='%(levelname)s %(asctime)s: %(name)s - %(message)s ',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

logger_main = logging.getLogger('main')
logger_cnn = logging.getLogger('env')
logger_att = logging.getLogger('agent')
logger_detailed = logging.getLogger('detailed')
logger_trading_info = logging.getLogger('trading_info')
logger_eval = logging.getLogger('eval')

# set the log levels for each logger
logger_main.setLevel(logging.INFO)
logger_cnn.setLevel(logging.DEBUG)
logger_att.setLevel(logging.DEBUG)
logger_detailed.setLevel(logging.INFO)
logger_trading_info.setLevel(logging.INFO)
logger_eval.setLevel(logging.INFO)

# set up the log file handlers
fh_main = logging.FileHandler('logs/log_main.txt')
fh_cnn = logging.FileHandler('logs/log_cnn.txt')
fh_att = logging.FileHandler('logs/log_att.txt')
fh_detailed = logging.FileHandler('logs/log_detailed.txt')
fh_trading_info = logging.FileHandler('logs/log_trading_info.txt')
fh_eval = logging.FileHandler('logs/log_eval.txt')

# set formatter when saving logs
formatter = logging.Formatter('%(levelname)s %(asctime)s: %(name)s - %(message)s ')
fh_main.setFormatter(formatter)
fh_cnn.setFormatter(formatter)
fh_att.setFormatter(formatter)
fh_detailed.setFormatter(formatter)
fh_trading_info.setFormatter(formatter)
fh_eval.setFormatter(formatter)

# add the handlers to the loggers
logger_main.addHandler(fh_main)
logger_cnn.addHandler(fh_cnn)
logger_att.addHandler(fh_att)
logger_detailed.addHandler(fh_detailed)
logger_trading_info.addHandler(fh_trading_info)
logger_eval.addHandler(fh_eval)