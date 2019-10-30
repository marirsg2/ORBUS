import numpy as np
import pickle
import sys
#--------------------------
target_file = "accumulated_results.results"
result_prefix = "scale_test_noise"
result_file_extension = ".result"
result_file_int = 1
result_block_size = 5
curr_results = []
all_results_string = ""
#BACKUP code do not remove
# start_capture_cases = ["BAYES ERROR LIST  [" , "LINEAR MODEL The average error"]
# stop_capture_and_reset_cases = [" ROUND  "]
# ignore_lines_cases = ["target and prediction"]


# also print out the in region MLE vs BAYES model IN REGION

start_capture_cases = ["BAYES ERROR LIST  [" ,]# "LINEAR MODEL The average error"]
stop_capture_and_reset_cases = []#[" ROUND  "]
ignore_lines_cases = ["target and prediction"]
default_std_out = sys.stdout
sys.stdout = open(target_file,'w')
noise_cases = [0.0,0.5,1.0,1.5]
try:
    while True:
        # todo update the code below to include noise in the string 0.0, 0.5, 1.0, 1.5
        result_file_name = result_prefix + str(1.0) + "_" + str(result_file_int) + result_file_extension
        #save the block_id, and results
        results_string = ""
        try:
            with open(result_file_name,"r") as result_file_handle:
                file_lines = result_file_handle.readlines()
                start_capture = False
                for line in file_lines:
                    for case in start_capture_cases:
                        if case in line:
                            start_capture = True
                    for case in stop_capture_and_reset_cases:
                        if case in line:
                            start_capture = False
                            results_string = ""
                    if start_capture and not any([x in line for x in ignore_lines_cases]) and len(line) < 500:
                        results_string += line
            #end with
            all_results_string += results_string
            result_file_int += 1
        except:
            break #out of while true loop

except:
    #no more results
    pass

print(all_results_string)