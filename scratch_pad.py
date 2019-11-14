import time
import datetime
import sys
import os

date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
date_time_str = date_time_str.replace(" ", "_")
date_time_str = date_time_str.replace("/", "_")
date_time_str = date_time_str.replace(",", "_")
date_time_str = date_time_str.replace(":", "_")
print("date and time:", date_time_str)
output_file_name = 'RBUS_output_results' + "_" + date_time_str + "1"
sys.stdout = open(output_file_name + '.txt', 'w')

print("just come text")

sys.stdout.flush()


date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
date_time_str = date_time_str.replace(" ", "_")
date_time_str = date_time_str.replace("/", "_")
date_time_str = date_time_str.replace(",", "_")
date_time_str = date_time_str.replace(":", "_")
print("date and time:", date_time_str)
output_file_name = 'RBUS_output_results' + "_" + date_time_str + "2"
sys.stdout = open(output_file_name + '.txt', 'w')

print("just come text")

sys.stdout.flush()


date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
date_time_str = date_time_str.replace(" ", "_")
date_time_str = date_time_str.replace("/", "_")
date_time_str = date_time_str.replace(",", "_")
date_time_str = date_time_str.replace(":", "_")
print("date and time:", date_time_str)
output_file_name = 'RBUS_output_results' + "_" + date_time_str + "3"
sys.stdout = open(output_file_name + '.txt', 'w')

print("just come text")

sys.stdout.flush()



