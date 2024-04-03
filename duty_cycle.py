import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import datetime, timedelta
from glob import glob

def filter_latest_versions(items):
    partition = {}
    for item in items:
        year_month_day = item.split('_')[-2]
        if year_month_day in partition.keys():
            partition[year_month_day].append(item)
        else:
            partition[year_month_day] = [item]
    latest_versions = []
    for key, value in partition.items():
        latest_versions.append(sorted(value)[-1])
    return sorted(latest_versions)

def get_data(paths):
    with open(paths[0], 'r') as f:
        headers = f.readlines()[9].split('\t')[1:]
        headers = ['date']+headers
    if type(paths) != list:
        paths = sorted([paths])
    data = pd.DataFrame()
    for idx, path in enumerate(paths):
        # if idx % (max(1,len(paths)//100)) == 0:
        #     print(f'{idx/len(paths)*100:.0f}%')
        _data = pd.read_csv(path, skiprows=23, sep='\t', names=headers)
        data = pd.concat([data, _data], ignore_index=True)
    data.date = data.date.map(lambda d: datetime.fromisoformat(d))
    return data

def get_switch_heater_group(pcdu_module, pcdu_group):
    return f'TCS_HtrGrp_{(pcdu_module-1)*6+pcdu_group:02d} HtrOn/Off'

def get_switch_state(lcl_bitwise_data, pcdu_module, pcdu_group, pcdu_thermal_switch):
    group = lcl_bitwise_data[get_switch_heater_group(pcdu_module, pcdu_group)]
    status = group.map(lambda g: int(f'{g:08b}'[::-1][pcdu_thermal_switch-1]))
    return status

def get_lcl_index(pcdu_module, pcdu_group, pcdu_thermal_switch):
    '''Get LCL heater index

    Get the LCL heater index. Note that this indexes from 1 not from 0.
    '''
    return ((pcdu_module-1)*6+pcdu_group-1)*8+pcdu_thermal_switch