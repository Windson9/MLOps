import sys
from src.logger import logging

def error_msg(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occured in python script [{0}] at line no. [{1}]: Error message - [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_msg=error_msg(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
    

    
