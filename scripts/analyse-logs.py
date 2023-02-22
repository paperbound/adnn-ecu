##########################################################################
#Input: Log files generated using the script 
#       perf-c-measure.py and perf-numpy-measure.py
#
#Output: dict: perf-data{"Network": [],"accuracy": [], "exec_time": []} 
#
##########################################################################
import os

os.chdir('../hdrnn/c-math.h/logs/')
result = {}
Network_Size = {'Test1': [784,2,10],
                 'Test2': [784,4,10],
                'Test3': [784,8,10],
                'Test4': [784,16,10],
                'Test5': [784,32,10],
                'Test6': [784,64,10],
                'Test7': [784,128,10],
                 'Test8': [784,256,10],
                 'Test9': [784,512,10],
                 'Test10': [784,1024,10],
                 'Test11': [784,32,16,10],
                 'Test12': [784,64,16,10],
                 'Test13': [784,128,16,10],
                 'Test14': [784,256,16,10],
                 'Test15': [784,512,16,10],
                 }

for trail_no in range(10):
    for key, value in Network_Size.items():
        pass