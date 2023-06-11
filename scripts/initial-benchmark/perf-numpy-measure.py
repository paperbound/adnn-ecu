#########################################################
#Performance measurements for the numpy implementation  #
#
#Instructions: Comment net.dump_weights("bias", "weights")
#               in run.py              
#########################################################
import os
path = 'python-numpy/'
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
                #  'Test11': [784,32,16,10],
                #  'Test12': [784,64,16,10],
                #  'Test13': [784,128,16,10],
                #  'Test14': [784,256,16,10],
                #  'Test15': [784,512,16,10],
                 }

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('logs/bias'):
    os.makedirs('logs/bias')
if not os.path.exists('logs/weights'):
    os.makedirs('logs/weights')


for i in range(1):
    if not os.path.exists('logs/trail' + str(i)):
        os.makedirs('logs/trail{0}'.format(i))
    if not os.path.exists('logs/bias/trail{0}'.format(i)):
        os.makedirs('logs/bias/trail{0}'.format(i))
    if not os.path.exists('logs/weights/trail{0}'.format(i)):
        os.makedirs('logs/weights/trail{0}'.format(i))
    
    for testid, nw_size in Network_Size.items():
        accr_fname = 'log_{0}_accr.txt'.format(testid)
        perf_fname = 'log_{0}_perf.txt'.format(testid)
        
        cmd = 'sed -i \'s/net = network.Network(.*/net = network.Network({0})/g\' run.py'.format(nw_size)
        os.system(cmd)
        
        cmd = '/usr/bin/time -v -o logs/trail{0}/{1} python3 run.py > logs/trail{0}/{2}'.format(i, perf_fname, accr_fname)
        os.system(cmd)
        
        cmd1 = 'mv weights logs/weights/trail{0}/weights-{1}'.format(i, testid)
        cmd2 = 'mv bias logs/bias/trail{0}/bias-{1}'.format(i, testid)
        os.system(cmd1 + '\n' + cmd2)