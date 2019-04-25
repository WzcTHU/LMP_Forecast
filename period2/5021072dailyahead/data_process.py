import pandas as pd

# import numpy as np

def process_gen_by_fuel_type(file_name):
    df = pd.read_csv(file_name)
    df = df.sort_index(ascending=False)
    group = df.groupby(df['fuel_type'])

    df2 = pd.DataFrame()

    i = 0
    for name, sub_group in group:
        sub_group = sub_group.reset_index()
        if (i == 0):
            df2['time'] = sub_group['datetime_beginning_ept']
            i += 1
        df2[name] = sub_group['mw']

    df2.to_csv('.\gen_by_fuel_type_20171001_20181101_processed.csv')

def process_load_frcstd_hist(file_name):
    df = pd.read_csv(file_name)
    group = df.groupby(df['forecast_hour_beginning_ept'])

    df2 = pd.DataFrame()
    i = 0
    for name, sub_group in group:
        df2 = df2.append(sub_group.iloc[-1:])
        # print(sub_group.iloc[-1:])
        # print(name, type(sub_group.iloc[-1:]))
    df2 = df2.sort_index()
    df2.to_csv('.\load_frcstd_hist_20171001_20181101_processed.csv', 
        columns=['forecast_hour_beginning_ept', 'forecast_load_mw'], index=False)

def proxess_hrl_load_prelim(file_name):
    df = pd.read_csv(file_name)
    df.to_csv('.\hrl_load_prelim_20171001_20181101_processed.csv', 
        columns=['datetime_beginning_ept', 'prelim_load_avg_hourly'], index=False)
    
def process_lmp(file_name):
    df = pd.read_csv('node5021072_lmp_20171001_20181101.csv')
    df.to_csv('.\\total_lmp_data_20171001_20181101_processed.csv', 
        columns=['datetime_beginning_ept', 'total_lmp_da'], index=False)
    df.to_csv('.\lmp_data_20171001_20181101_processed.csv', 
        columns=['datetime_beginning_ept', 'total_lmp_da', 'system_energy_price_da', 
        'congestion_price_da', 'marginal_loss_price_da'], index=False)

def process_outage(file_name):
    df = pd.read_csv(file_name)
    df = df.sort_index(ascending=False)
    group = df.groupby(df['region'])
    df2 = group.get_group('PJM RTO')
    df3 = pd.DataFrame()
    # print(type(df2.iloc[2]))
    # input()
    for i in range(0, len(df2)):
        if (df2.loc[df2.index[i], 'forecast_execution_date_ept'] == df2.loc[df2.index[i], 'forecast_date']):
            df3 = df3.append(df2.iloc[i])
    print(df3)
    df3.to_csv('.\gen_outage_20171001_20181101_processed.csv', index=False, columns=['forecast_date', 'total_outages_mw'])
    # for name, sub_group in group:
    #     print(name, type(sub_group))


if __name__ == '__main__':
    # process_gen_by_fuel_type('gen_by_fuel_type_20171001_20181101.csv')
    # process_load_frcstd_hist('load_frcstd_hist_AEP_20171001_20181101.csv')
    # proxess_hrl_load_prelim('hrl_load_prelim_AEP_20171001_20181101.csv')
    # process_lmp('node5021072_lmp_20171001_20181101.csv')
    process_outage('gen_outages_by_type_20171001_20181101.csv')

    