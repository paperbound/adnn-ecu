#***************ALGORITHM****************#
#PARAMETERS: String that modifies No.of Neurons
#            Output dictionary with TestNo, execution time and accuracy
#            String for Log file base name
#            
#Flow:  Set log file name
#       Set number of neurons in network.c file
#       Run the Perf command and save the output into a log file
#       Repeat for different Hidden layer sizes
#
#Functions:     modify-network-size([network_size])
#               
##########################################
import os
import re

os.chdir('../hdrnn/c-math.h')
Network_Size = {'Test1': [784,2,10],
                'Test2': [784,4,10],
                'Test3': [784,8,10],
                'Test4': [784,16,10],
                'Test5': [784,32,10],
                'Test6': [784,64,10],
                'Test7': [784,128,10],
                 'Test8': [784,256,10],
                 'Test9': [784,512,10],
                 'Test10': [784,1024,10]}

def modify_network_size(size):
    if len(size) > 3:
        #TODO Implement changes to source files
        print('Hidden Layer 2 Size: ' + size[2])
        
    else:
        #TODO Implement changes for multiple hidden layers
        cmd = 'sed -i \'s/#define HIDDEN_LAYER.*/#define HIDDEN_LAYER_SIZE {0}/g\' src/network.h'.format(size[1])
        os.system(cmd)
        os.system('make')

for testid, nw_size in Network_Size.items():
    nw_Output_Filename = 'log_{0}_nw.txt'.format(testid)
    perf_Output_Filename = 'log_{0}_perf.txt'.format(testid)
    modify_network_size(nw_size)
    cmd = 'sudo perf stat -o logs/{0} -- ./bin/hdrnn -train 100 -weights weights/weights-{2} -bias bias/bias-{2} > logs/{1}'.format(perf_Output_Filename, nw_Output_Filename, testid)
    os.system(cmd)