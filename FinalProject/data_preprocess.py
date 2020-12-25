# -*- coding: UTF-8 -*-
import os
from datetime import datetime, date, time, timezone, timedelta

input_path ='./20years'
outputdirtrain = './20clean_train1'

if not os.path.isdir(outputdirtrain):
    os.mkdir(outputdirtrain)
outputdirtest = './20clean_test'
if not os.path.isdir(outputdirtest):
    os.mkdir(outputdirtest)


enddate = '2016-12-31'

for filename in os.listdir(input_path):
    with open( os.path.join(input_path, filename) ,'r') as fin:
        lines = fin.readlines()
        content_train = lines[0]
        content_test = lines[0]
        for line in lines[1:]:
            if 'null' in line:
                continue
            A = datetime.strptime(line.split(',')[0], '%Y-%m-%d')
            B = datetime.strptime(enddate, '%Y-%m-%d')
            if A > B:
                content_test += line
            else:
                content_train += line
        
        with open(os.path.join(outputdirtrain, filename), 'w') as fout:
            fout.write(content_train)
        with open(os.path.join(outputdirtest, filename), 'w') as fout:
            fout.write(content_test)


