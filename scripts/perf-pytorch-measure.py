#########################################################
#Performance measurements for the numpy implementation  #
#
#Instructions: Comment net.dump_weights("bias", "weights")
#               in run.py              
#########################################################
import os
path = 'libtorch/'
os.chdir(path)
Network_Size = {'Test1': [784,2,10],
                # 'Test2': [784,4,10],
                 'Test3': [784,8,10],
                # 'Test4': [784,16,10],
                'Test5': [784,32,10],
                # 'Test6': [784,64,10],
                'Test7': [784,128,10],
                #  'Test8': [784,256,10],
                #  'Test9': [784,512,10],
                #  'Test10': [784,1024,10],
                  'Test11': [784,32,16,10],
                #  'Test12': [784,64,16,10],
                #   'Test13': [784,128,16,10],
                #  'Test14': [784,256,16,10],
                #  'Test15': [784,512,16,10],
                }

if not os.path.exists('build'):
    os.makedirs('build')
if not os.path.exists('logs'):
    os.makedirs('logs')

for i in range(10):
    if not os.path.exists('logs/trail' + str(i)):
        os.makedirs('logs/trail{0}'.format(i))
    
    for testid, nw_size in Network_Size.items():
        accr_fname = 'log_{0}_accr.txt'.format(testid)
        perf_fname = 'log_{0}_perf.txt'.format(testid)
        
        if len(nw_size) == 3:
            #Editing the first part
            cmd1 = 'sed -i \'s/fc1 = register_modul.*/fc1 = register_module("fc1", torch::nn::Linear(784, ' + str(nw_size[1]) + '));/g\' build/mnist.cpp'
            cmd2 = 'sed -i \'s/fc2 = register_modul.*/fc2 = register_module("fc2", torch::nn::Linear(' + str(nw_size[1]) + ', 10));/g\' build/mnist.cpp'
            cmd3 = 'sed -i \'s/fc3 = register_modul.*//g\' build/mnist.cpp'
            
            #Editing the second part
            cmd4 = 'sed -i \'s/x = torch::sigmoid(fc3.*//g\' build/mnist.cpp'

            #Editing the third part
            cmd5 = 'sed -i \'s/torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3.*/torch::nn::Linear fc1{nullptr}, fc2{nullptr};/g\' build/mnist.cpp'
            
            os.system(cmd1 + '\n' + cmd2 + '\n' + cmd3 + '\n' + cmd4 + '\n' + cmd5)

        elif len(nw_size) == 4:
            #Editing the first part
            cmd1 = 'sed -i \'s/fc1 = register_modul.*/fc1 = register_module("fc1", torch::nn::Linear(784, ' + str(nw_size[1]) + '));/g\' build/mnist.cpp'
            cmd2 = 'sed -i \'s/fc2 = register_modul.*/fc2 = register_module("fc2", torch::nn::Linear(' + str(nw_size[1]) + ', ' + str(nw_size[2]) + '));/g\' build/mnist.cpp'
            cmd3 = 'sed -i \'/fc2 = register_modul.*/a fc3 = register_module("fc3", torch::nn::Linear(' + str(nw_size[2]) + ', 10));\' build/mnist.cpp'

            #Editing the second part
            cmd4 = 'sed -i \'/x = torch::sigmoid(fc2.*/a x = torch::sigmoid(fc3->forward(x));\' build/mnist.cpp'

            #Editing the third part
            cmd5 = 'sed -i \'s/torch::nn::Linear fc1{nullptr}, fc2.*/torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};/g\' build/mnist.cpp'

            os.system(cmd1 + '\n' + cmd2 + '\n' + cmd3 + '\n' + cmd4 + '\n' + cmd5)

        else:
            print('Error while editing source file')

        cmd1 = 'cd build'
        cmd2 = 'make'
        os.system(cmd1 + '\n' + cmd2)
        cmd = 'cd ..'
        os.system(cmd)        
        cmd = '/usr/bin/time -v -o logs/trail{0}/{1} ./build/mnist > logs/trail{0}/{2}'.format(i, perf_fname, accr_fname)
        os.system(cmd)
        
        cmd = 'mv net.pt logs/trail{0}/log_{1}_model_param.pt'.format(i, testid)
        os.system(cmd)