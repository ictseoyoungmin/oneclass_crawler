import RunCrawl as rc
import file_move_py as fm
from model import Train

import sys
import os

key = ''
base_path = ''

if __name__ == '__main__':
    
    # "python main.py cat"
    argument = sys.argv
    del argument[0]
    key = argument[0]       # class
    base_path = os.getcwd()
    
    # Crawling - 비동기 작업 1
    runc = rc.Crawl()
    runc.set_para(key,base_path)
    # runc.run()
    print('Crawling done.')
    
    # model training - 비동기 작업 2
    # model.py 개발필요
    runt = Train()
    runt.set_para(key,base_path)
    #runt.run()
    print('Training done.')
    
    # Classification + file move : Psedo-labeling
    runm = fm.FileMove()
    parac = runc.get_para()
    parat = runt.get_para() # model_path + model_name runc.run()미사용시 밑 주석해제
    parat = parat+f'model_test.h5'
    runm.set_para(parac[0],parac[1],parac[2],parat)
    runm.run()
    print('Pseudo labeling done.')
    
    # Classfication in abnormal folder + file move
    runt.set_para(key,base_path,main_store_path=runm.get_para()) # /data/key_result/
    print('train : ',runt.train_path)
    runt.run2()
    print('Training done.')

    runm.set_second_target() # target_path : [crawl_path => normal ]-> [abnormal_path => normal]
    runm.run()

    print('Program done.')