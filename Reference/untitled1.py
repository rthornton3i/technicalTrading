import multiprocessing as mp
from multiprocessing import Process,Manager,Queue
from time import time
import random as r

def foo(vals,d):
    test = []
    for _ in range(int(1e3*d)):
        a = r.randint(2,10) ** r.randint(2,6)
        test.append(a)
   
    vals.put(test)
    return vals

def setManager(mng):
    vals = mng.Namespace()
    vals.test = mng.list()
    
    return vals

if __name__ == '__main__':
    print('start')
    tic = time()
    # foo(2,8,8)
    toc = time() - tic
    print('Single core time: ' +str(toc))
    
    
    tic = time()
    # mng = Manager()
    # vals = setManager(mng)
    vals = Queue()
    
    processes = []
    for i in range(mp.cpu_count()):
        processes.append(Process(target=foo, args=(vals,1)))
     
    print('starting')
    [process.start() for process in processes] 
    print('joining')       
    [process.join() for process in processes]
    print('getting')
    # results = [process.get() for process in processes]
    results = vals.get()
   
    # results = [res.get(timeout=1) for res in multiple_results]
    
    toc = time() - tic
    print('Multi-core time: ' + str(toc))