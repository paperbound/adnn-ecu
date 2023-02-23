##########################################################################
#Input: Log files generated using the script 
#       perf-c-measure.py and perf-numpy-measure.py
#
#Output: dict: perf-data{[<"Network">], [<"accuracy">], [<"exec_time">]} 
#
##########################################################################
import os
import re

# os.chdir('../hdrnn/c-math.h/logs/')
c_path = '../hdrnn/c-math.h/'
numpy_path = ('../hdrnn/python-numpy/')

c_accr_pattern = 'Network classified \d*'
numpy_accr_pattern = 'Epoch 29: \d*'
perf_pattern = '\d*.\d* seconds time elapsed'

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

result = {"Network": [0]*len(Network_Size),
        "Accuracy": [0]*len(Network_Size),
        "ExecTime": [0]*len(Network_Size)}

def calc_nparams(shape):
    inputs = shape[0]
    # np = inputs
    np = 0
    for l in shape[1:]:
        np += (l + inputs * l)
        inputs = l
    return np

def get_accr(trail_no, key, impl_type):
    if impl_type == 'c':
        path = c_path
        pattern = c_accr_pattern
        filename = path + 'logs/trail{0}/log_{1}_nw.txt'.format(trail_no, key)
    else:
        path = numpy_path
        pattern = numpy_accr_pattern
        filename = path + 'logs/trail{0}/log_{1}_accr.txt'.format(trail_no, key) 
    f = open(filename, 'r')
    matches = re.findall(r'{0}'.format(pattern), f.read())
    accuracy = int(re.search(r'\d\d\d\d', matches[-1]).group()) / 100
    return accuracy

def get_exec_time(trail_no, key, impl_type):
    if impl_type == 'c':
        path = c_path
    else:
        path = numpy_path
    f = open(path + 'logs/trail{0}/log_{1}_perf.txt'.format(trail_no, key), 'r')    
    matches = re.findall(r'{0}'.format(perf_pattern), f.read())
    exec_time = float(re.search(r'\d*.\d*', matches[-1]).group())
    print(exec_time)
    return exec_time

impl_type = ''
while (impl_type != 'c') and (impl_type != 'numpy'):
    impl_type = input('Enter Implementation type to collect logs (Option: c or numpy):')

for trail_no in range(1):
    for index, (key, value) in enumerate(Network_Size.items()):
        result["Network"][index] = calc_nparams(value)
        result['Accuracy'][index] += get_accr(trail_no, key, impl_type)
        result['ExecTime'][index] += get_exec_time(trail_no, key, impl_type)


result['Accuracy'] = [round((i / (trail_no + 1)),2) for i in result['Accuracy']]
result['ExecTime'] = [round((i / (trail_no + 1)),2) for i in result['ExecTime']]

print(result)