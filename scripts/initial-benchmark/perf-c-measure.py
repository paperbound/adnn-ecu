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

os.chdir('c-math.h')

Network_Size = {'Test1': [784,2,10],
                 'Test2': [784,4,10],
                'Test3': [784,8,10],
                #'Test4': [784,16,10],
                'Test5': [784,32,10],
                #'Test6': [784,64,10],
                'Test7': [784,128,10],
                # 'Test8': [784,256,10],
                # 'Test9': [784,512,10],
                # 'Test10': [784,1024,10],
                 'Test11': [784,32,16,10],
                # 'Test12': [784,64,16,10],
                # 'Test13': [784,128,16,10],
                # 'Test14': [784,256,16,10],
                # 'Test15': [784,512,16,10],
                 }

def modify_network_size(size):
    if len(size) > 3:
        #define modifications
        cmd0 = 'sed -i \'/#define HIDDEN_LAYER_SIZE2.*/d\' src/network.h'
        cmd1 = 'sed -i \'s/#define HIDDEN_LAYER_SIZE.*/#define HIDDEN_LAYER_SIZE1 {0}/g\'\
         src/network.h'.format(size[1])
        cmd2 = 'sed -i \'/^#define HIDDEN_LAYER_SIZE1.*/a #define HIDDEN_LAYER_SIZE2 {0}\'\
         src/network.h'.format(size[2])
        os.system(cmd0 + '\n' + cmd1 + '\n' + cmd2)
        
        #add_layer modifications
        cmd3 = 'sed -i \'/add_layer(network, HIDDEN_LAYER_SIZE2.*/d\' src/network.h'
        cmd4 = 'sed -i \'s/add_layer(network, HIDDEN_LAYER_SIZE.*/\
add_layer(network, HIDDEN_LAYER_SIZE1, INPUT_LAYER_SIZE);/g\'\
                 src/network.h'
        cmd5 = 'sed -i \'s/add_layer(network, OUTPUT_LAYER_SIZE.*/\
add_layer(network, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE2);/g\'\
                 src/network.h'
        cmd6 = 'sed -i \'/add_layer(network, HIDDEN_LAYER_SIZE1.*/a\
             add_layer(network, HIDDEN_LAYER_SIZE2, HIDDEN_LAYER_SIZE1);\'\
                 src/network.h'
        os.system(cmd3 + '\n' + cmd4 + '\n' + cmd5 + '\n' + cmd6)

    else:
        cmd0 = 'sed -i \'/#define HIDDEN_LAYER_SIZE2.*/d\' src/network.h'
        cmd1 = 'sed -i \'s/#define HIDDEN_LAYER.*/#define HIDDEN_LAYER_SIZE {0}/g\'\
         src/network.h'.format(size[1])
        
        cmd2 = 'sed -i \'s/add_layer(network, HIDDEN_LAYER_SIZE1.*/\
add_layer(network, HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);/g\'\
                 src/network.h'
        cmd3 = 'sed -i \'s/add_layer(network, OUTPUT_LAYER_SIZE.*/\
add_layer(network, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);/g\'\
                 src/network.h'
        cmd4 = 'sed -i \'/add_layer(network, HIDDEN_LAYER_SIZE2.*/d\'\
             src/network.h'
        os.system(cmd0 + '\n' + cmd1 + '\n' + cmd2 + '\n' + cmd3 + '\n' + cmd4)
    os.system('make')


for i in range(2):
    os.system('mkdir logs/trail{0}'.format(i))
    os.system('mkdir bias/trail{0}'.format(i))
    os.system('mkdir weights/trail{0}'.format(i))
    for testid, nw_size in Network_Size.items():
        nw_Output_Filename = 'log_{0}_nw.txt'.format(testid)
        perf_Output_Filename = 'log_{0}_perf.txt'.format(testid)
        modify_network_size(nw_size)
        cmd = '/usr/bin/time -v -o logs/trail{3}/{0} ./bin/hdrnn -train 100 \
            -weights weights/trail{3}/weights-{2} -bias bias/trail{3}/bias-{2} > \
            logs/trail{3}/{1}'.format(perf_Output_Filename, nw_Output_Filename, testid, i)
        os.system(cmd)