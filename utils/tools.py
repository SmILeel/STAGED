import datetime
import time
import numpy as np
import pandas as pd
import torch


def from_graph_num_to_time(win_num, win_size, slide_stride, down_sample=10, temporal_length=4, dataset='swat'):
    if dataset == 'swat':
        start_time = datetime.datetime(year=2015, month=12, day=28, hour=10, minute=0, second=0)
        time_delta = datetime.timedelta(seconds=(((temporal_length+win_num)-1)*slide_stride+win_size)*10)
        given_time = start_time + time_delta
        str_time = given_time.strftime('%Y-%m-%d %H:%M:%S')
        return str_time

def from_time_to_graph_num(str_time, win_size, slide_stride, down_sample, temporal_length=4, dataset='swat'):
    if dataset == 'swat':
        start_time = datetime.datetime(year=2015, month=12, day=28, hour=10, minute=0, second=0)
        given_time = datetime.datetime.strptime(str_time, '%Y/%m/%d %H:%M:%S')
        time_delta = (given_time - start_time).seconds
        num_win = ((time_delta//10)-win_size)//slide_stride
        return num_win


