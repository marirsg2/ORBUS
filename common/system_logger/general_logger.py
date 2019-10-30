import logging

#Create and configure logger
logging.basicConfig(format='%(asctime)s Module - %(module)s - Function - %(funcName)s(): %(message)s',
                    datefmt='%H:%M:%S')

#Creating an object
logger=logging.getLogger()
logger.setLevel(logging.INFO)