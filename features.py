from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

import os

import helpers as h
import duty_cycle as dc

from glob import glob

# get orbit
import astrospice
import astropy.units as u
from sunpy.coordinates import HeliographicCarrington
import os

import argparse
import logging

carr_frame = HeliographicCarrington(observer='self')

solo_kernels = astrospice.registry.get_kernels('solar orbiter', 'predict')
solo_kernel = solo_kernels[0]
solo_coverage = solo_kernel.coverage('SOLAR ORBITER')

RDS_HK_FOLDER = "/rds/general/user/rl4215/projects/solarorbitermagnetometer/live/production"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--start',
                        '-s',
                        default=datetime(2020,3,1),
                        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
                        help='Specify the start in the format YYYY-MM-DD')
    
    parser.add_argument('--end',
                        '-e',
                        default=datetime(2024,1,24),
                        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
                        help='Specify the end in the format YYYY-MM-DD')
    
    # Define an argument for the boolean variable
    parser.add_argument('--plot', '-p', action='store_true', help='To plot figures or not')

    parser.add_argument(
        "--history",
        type=int,
        default=3,
        help="How far back to include as features",
    )

    args = parser.parse_args()
    plot = args.plot
    ###############################################################################
    ### Radius ####################################################################
    ###############################################################################
    
    # Radius and Heater
    n_days_back = args.history
    start = args.start - timedelta(days = n_days_back)
    end = args.end

    all_dates = h.get_dates(start, end)

    # Adds the distance from the Sun as a feature
    coords = astrospice.generate_coords('SOLAR ORBITER', all_dates)
    solo_coords = coords.transform_to(carr_frame)
    radii = solo_coords.radius.to(u.au).value


    # features = pd.DataFrame({
    #     "Date": all_dates,
    #     "Radius": radii,
    #     })
    features = pd.DataFrame({
        "Date": all_dates,
        })

    #! could add distance to Earth or Venus?

    # This step is very important!
    # reduce cardinality of radius, which is fancy speak for only have N_radii number of values that Radius can take
    # Previously, because I got the exact radius per day, it meant that the model could just learn that such a value
    # corresponds to June 5th, and it knows the profiles near this date, so just copies them.
    # Where I actually want to see the true changes because of Radius, not because of the pseudo-date that it works out
    # #! But LightGBM does this itself, I think?
    
    
    
    N = 100
    features['Radius'] = radii
    features = h.reduce_cardinality(features, 'Radius', N)
    
    if plot:
        fig, axs = plt.subplots(1,1,sharex = True)
        
        axs.plot(features['Radius'])
        axs.set_ylabel('Radius')
        
        fig.autofmt_xdate()
        
        fig.savefig('Figures/radius.png')
    
    ######################################################################################
    ### Distance f earth ###################################################################
        
    coords_earth = astrospice.generate_coords('EARTH', all_dates)  # Adjust this line based on your actual function for generating coordinates
    earth_coords = coords_earth.transform_to(carr_frame)

    # Calculate distances
    radii_solo = solo_coords.radius.to(u.au).value
    distance_to_earth = solo_coords.separation_3d(earth_coords).to(u.au).value  # Calculate 3D separation in astronomical units


    N = 100
    features['DistanceToEarth'] = distance_to_earth
    features = h.reduce_cardinality(features, 'DistanceToEarth', N)  # You might want to adjust or replicate this for 'DistanceToEarth' if needed

    # Plotting
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)  # Create two subplots for Radius and Distance to Earth
        axs[0].plot(features['Date'], features['Radius'])
        axs[0].set_ylabel('Radius (AU)')
        
        axs[1].plot(features['Date'], features['DistanceToEarth'])
        axs[1].set_ylabel('Distance to Earth (AU)')
        
        fig.autofmt_xdate()
        
        fig.savefig('Figures/radius_and_distance_to_earth.png')

    ######################################################################################
    ### Distance f venus ###################################################################
    coords_venus = astrospice.generate_coords('VENUS', all_dates)  # Adjust this line based on your actual function for generating coordinates
    venus_coords = coords_venus.transform_to(carr_frame)

    # Calculate distances
    radii_solo = solo_coords.radius.to(u.au).value
    distance_to_venus = solo_coords.separation_3d(venus_coords).to(u.au).value  # Calculate 3D separation in astronomical units


    N = 100
    features['DistanceToVenus'] = distance_to_venus
    features = h.reduce_cardinality(features, 'DistanceToVenus', N)  # You might want to adjust or replicate this for 'DistanceToEarth' if needed

    # Plotting
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)  # Create two subplots for Radius and Distance to Earth
        axs[0].plot(features['Date'], features['Radius'])
        axs[0].set_ylabel('Radius (AU)')
        
        axs[1].plot(features['Date'], features['DistanceToVenus'])
        axs[1].set_ylabel('Distance to Venus (AU)')
        
        fig.autofmt_xdate()
        
        fig.savefig('Figures/radius_and_distance_to_venus.png')
    
    # original code if reduce)cardinality doesnt work
    # N_radii = 100
   # # r_bins = np.linspace(features['Radius'].min(), features['Radius'].max(), N_radii)
   # # features['Radius_bin'] = np.digitize(features['Radius'], bins = r_bins) 
   # # # so just radius_bin as the feature, the model doesn't need to know its true value
   # # features.drop(['Radius'], axis = 1, inplace = True)
   # 
   # ###############################################################################
   # ### Heater 1 or 2 #############################################################
   # ###############################################################################
   # # encode the fact that the heaters changed on November 8th 2021. 
   # # This is important for the decaying Bt signal
   # # features['Heater'] = 1
   # # features.loc[features['Date'] > datetime(2021,11,8), 'Heater'] = 2
    
    ###############################################################################
    ### th_mag ####################################################################
    ###############################################################################

    print("TH MAG")
    print('')

    # I need to load all the data like normal, then decide how I can summarise that per day
    # if start < datetime(2020,6,22):
    #     # if you get this working then downsample to every 10 seconds and save
    #     raise Exception("not sorted out what to do with these big files yet, choose a later start time")

    th_mag = h.load_HK("solo_ANC_sc-thermal-mag", start, end)
    th_mag_per_day = th_mag.resample("1D").median(numeric_only=True)
    
    # print(features.index)
    # print(th_mag_per_day.index)
    
    features['OBS T'] = th_mag_per_day["ANP_2_1_6 MAG OBS"].values
    features['IBS T'] = th_mag_per_day["ANP_2_1_9 MAG IBS"].values
    features['OBS minus IBS T'] = th_mag_per_day["ANP_2_1_6 MAG OBS"].values - th_mag_per_day["ANP_2_1_9 MAG IBS"].values
    
    # time since dropped
    encoded_temp = np.where(th_mag_per_day["ANP_2_1_6 MAG OBS"].values > -70, 1,0)

    time_since = [0]
    time_it_was_high = [0]
    cumulative = [0]

    cu_val = 0
    length_high_temp = 0
    days_since_drop = 0

    for i in np.arange(1, encoded_temp.shape[0]):
        previous = encoded_temp[i-1]
        now = encoded_temp[i]
        if now > previous:
            # jump in temp
            #this takes care of just up and down I think
            length_high_temp = 1
            cu_val += 1
            
        
        elif now < previous:
            # drop in temp
            days_since_drop = 0
        else:
            # then values must be the same
            if now > 0.5:
                # we are in a high temp bit
                length_high_temp += 1
                cu_val += 1
            else:
                days_since_drop += 1
                if cu_val > 0:
                    cu_val -= 1

        time_since.append(days_since_drop)
        time_it_was_high.append(length_high_temp)
        cumulative.append(cu_val)

    features['time_since_T_high'] = time_since
    # I don't think I need this now, cumulative models it better
    features['time_it_was_high'] = time_it_was_high
    features['cumulative'] = cumulative
    
    N = 20
    for feature_name in ['OBS T', 'IBS T', 'OBS minus IBS T']:
        features = h.reduce_cardinality(features, feature_name, N)
    
    if plot:
        fig, axs = plt.subplots(6,1,sharex=True)

        axs[0].scatter(th_mag.index, th_mag["ANP_2_1_6 MAG OBS"])
        axs[0].scatter(th_mag.index, th_mag["ANP_2_1_9 MAG IBS"])
        axs[0].set_ylabel('T')
        
        axs[1].scatter(features['Date'], features["OBS T"])
        axs[1].scatter(features['Date'], features["IBS T"])
        axs[1].set_ylabel('T per day')
        axs[1].yaxis.set_label_position("right")
        axs[1].yaxis.tick_right()
        
        axs[2].scatter(features['Date'], features["OBS minus IBS T"])
        axs[2].set_ylabel('OBS - IBS')
        
        axs[3].scatter(features['Date'], features['time_since_T_high'])
        axs[3].set_ylabel('time_since_T_high')
        axs[3].yaxis.set_label_position("right")
        axs[3].yaxis.tick_right()
        
        axs[4].scatter(features['Date'], features['time_it_was_high'])
        axs[4].set_ylabel('time_it_was_high')
        

        axs[5].scatter(features['Date'], features['cumulative'])
        axs[5].set_ylabel('Cumulative')
        axs[5].yaxis.set_label_position("right")
        axs[5].yaxis.tick_right()
        fig.autofmt_xdate()
        
        
        fig.savefig("Figures/th_mag.png")
   
    ###############################################################################
    ### Duty cycle ################################################################
    ###############################################################################
    print("Duty cycle")
    print('')
    
    if start < datetime(2021,1,1):
        raise Exception("No data before 2021")

    lcl_bitwise_paths_ = dc.filter_latest_versions(glob(h.RDS_FOLDER + 'solo_ANC_sc-lcl-bitwise-mag/2021/*.txt'))
    lcl_bitwise_paths_.extend(dc.filter_latest_versions(glob(h.RDS_FOLDER + 'solo_ANC_sc-lcl-bitwise-mag/2022/*.txt')))
    lcl_bitwise_paths_.extend(dc.filter_latest_versions(glob(h.RDS_FOLDER + 'solo_ANC_sc-lcl-bitwise-mag/2023/*.txt')))
    lcl_bitwise_paths_.extend(dc.filter_latest_versions(glob(h.RDS_FOLDER + 'solo_ANC_sc-lcl-bitwise-mag/2024/*.txt')))
    lcl_bitwise_paths = []
    for path in lcl_bitwise_paths_:
        date_str = path[-16:-8]
        date = datetime.strptime(date_str, "%Y%m%d")
        if date >= start and date <= end:
            lcl_bitwise_paths.append(path)
            
    lcl_bitwise_data = dc.get_data(lcl_bitwise_paths)
    
    time = lcl_bitwise_data.date
    htr_1_5_5 = dc.get_switch_state(lcl_bitwise_data, 1, 5, 5)
    htr_1_5_5.index = time
    htr_2_5_5 = dc.get_switch_state(lcl_bitwise_data, 2, 5, 5)
    htr_2_5_5.index = time

    # Calculate the duty cycle from a 1 hour rolling mean
    htr_1_5_5_dutycycle = 100*htr_1_5_5.rolling(timedelta(hours=1)).mean()
    htr_2_5_5_dutycycle = 100*htr_2_5_5.rolling(timedelta(hours=1)).mean()

    # Instead of a 1 hour rolling mean, resample the data to 1 hour and take the mean
    # This can get rid of the aliasing that can occur with the rolling mean
    htr_1_5_5_dutycycle = htr_1_5_5_dutycycle.resample('1H').mean()
    htr_2_5_5_dutycycle = htr_2_5_5_dutycycle.resample('1H').mean()
    
    ht1 = htr_2_5_5_dutycycle.resample('1D').max()
    ht2 = htr_1_5_5_dutycycle.resample('1D').max()

    dutycycle = ht1.combine(ht2, max)
    
    features['dutycycle'] = dutycycle.values
    features = h.reduce_cardinality(features, 'dutycycle', 50)
    
    if args.plot:
        fig, axs = plt.subplots(1,1,sharex=True)

        axs.plot(features['Date'], features['dutycycle'])
        axs.set_ylabel('Duty cycle')
        
        fig.autofmt_xdate()
        fig.savefig("Figures/dutycycle.png")

    ###############################################################################
    ### th_swa_rpw ################################################################
    ###############################################################################
    
    print('TH SWA RPW')
    print('')
    
    th_swa_rpw = h.load_HK('solo_ANC_sc-thermal-swa-rpw', start, end)
    
    
    th_swa_rpw_day = th_swa_rpw.resample('1D').mean()
    
    features['SCM T'] = th_swa_rpw_day['ANP_1_1_7 RPW SCM SRP'].values
    features['SCM T grad'] = np.gradient(features['SCM T'].values)
    
    bins = np.linspace(features['SCM T'].min(),-103,20)
    bins = np.concatenate((bins, np.array([-50, 100])))
    features = h.reduce_cardinality(features, 'SCM T', 0, bins = bins)
    
    bins = np.arange(-5,5+0.5,0.5)
    bins = np.concatenate((np.array([-100]), bins, np.array([100])))
    features = h.reduce_cardinality(features, 'SCM T grad', N, bins = bins)
    
    
    if args.plot:
        fig, axs = plt.subplots(2,1,sharex=True)
        
        # axs.plot(th_swa_rpw.index , th_swa_rpw['ANP_1_1_7 RPW SCM SRP'])
        
        axs[0].plot(features['Date'], features['SCM T'])
        axs[0].set_ylabel('SCM T')
        
        axs[1].plot(features['Date'], features['SCM T grad'])
        
        fig.savefig("Figures/RPW.png")
    '''
    #########################################################################
    ##LCL #######################################################################
    ###############################################################################
    
    print('LCL')
    print('')
    lcl_to_save = pd.read_csv('HK data/mag_lcl.csv', parse_dates = ['Time'])
    lcl_to_save.set_index('Time', inplace=True)

    lcl_to_save = lcl_to_save.loc[(lcl_to_save.index >= start) & (lcl_to_save.index <= end + timedelta(days = 1))]
    
    lcl_PWR_sum = lcl_to_save['B_LCL2_16 MAG-B PWR TM'].resample('1D').mean()#.apply(lambda x: h.mode(x, 20))
    lcl_day_A_mode = lcl_to_save['B_LCL4_17 MAG-A TM'].resample('1D').apply(lambda x: h.mode(x, 20))
    
    A_on_off  = np.zeros(lcl_to_save.shape[0])

    threshold = 0.1
    A_on_off = np.where(lcl_to_save['B_LCL4_17 MAG-A TM'].values  < threshold, 1, A_on_off)
    lcl_to_save['A_on_off'] = A_on_off
    A_on_off_day = lcl_to_save['A_on_off'].resample('1D').mean()
    A_on_off_day_result = np.where(A_on_off_day.values  > 0, np.log(A_on_off_day.values), A_on_off_day.values)
    
    lcl_PWR_sum_aligned = lcl_PWR_sum.reindex(features.index, fill_value=np.nan)
    lcl_day_A_mode_aligned = lcl_day_A_mode.reindex(features.index, fill_value=np.nan)
    A_on_off_day_result_aligned = pd.Series(A_on_off_day_result, index=lcl_PWR_sum.index).reindex(features.index, fill_value=np.nan)

    # Now you can safely assign without length mismatch
    features['Mag PWR'] = lcl_PWR_sum_aligned
    features['Mag A'] = lcl_day_A_mode_aligned
    features['No time A off'] = A_on_off_day_result_aligned
    
    bins = np.linspace(0, 0.1, 20)
    bins = np.concatenate((bins, np.array([10])))
    features = h.reduce_cardinality(features, 'Mag PWR', 0, bins = bins)

    bins = np.linspace(0.25211, features['Mag A'].max(), 20)
    bins = np.concatenate((np.array([-1]), bins))
    features = h.reduce_cardinality(features, 'Mag A', 0, bins = bins)

    features = h.reduce_cardinality(features, 'No time A off', 10)
    
    if args.plot:

        fig, axs = plt.subplots(3,1,sharex=True)

        axs[0].plot(features['Date'] , features['Mag PWR'][:len(features['Date'])])
        axs[0].set_ylabel('Mag PWR')
        axs[1].plot(features['Date'] , features['Mag A'])
        axs[1].set_ylabel('Mag A')
        axs[1].yaxis.set_label_position("right")
        axs[1].yaxis.tick_right()
        axs[2].plot(features['Date'] , features['No time A off'])
        axs[2].set_ylabel('No times A off')
        fig.autofmt_xdate()
        fig.savefig('Figures/lcl.png')
    '''
    
    ###############################################################################
    ### SA ########################################################################
    ###############################################################################

    print('SA')
    print('')

    
    sa = h.load_HK('solo_ANC_sc-solar-arrays', start, end)
    
    sade_my = sa['SADE A MY CalPosSensor'].resample('1D').min()
    
    threshold = 0.2

    def sa_encode(arr):
        # get the unique values 
        
        unique = np.unique(arr)
        
        # just 0s or 1
        if 1 in unique and -1 in unique:
            return 2
        elif 1 in unique:
            return 1
        elif -1 in unique:
            return -1
        else:
            return 0


    # plt.figure()
    # sa_diff = np.where(sa_diff > threshold, 1, sa_diff)
    # sa_diff = np.where(sa_diff < -threshold, -1, sa_diff)
    # sa_diff = np.where((sa_diff > -threshold) & (sa_diff < threshold), 0, sa_diff)
    # plt.plot(sa_diff)


    # smooth out the data so the diff works better
    sa_1min = sa.resample('1T').max()
    sa_diff = np.gradient(sa_1min['SADE A MY CalPosSensor'])

    sa_diff = np.where(sa_diff > threshold, 1, sa_diff)
    sa_diff = np.where(sa_diff < -threshold, -1, sa_diff)
    sa_diff = np.where((sa_diff > -threshold) & (sa_diff < threshold), 0, sa_diff)

    sa_1min['diff'] = sa_diff

    sa_1day = sa_1min['diff'].resample('1D').apply(sa_encode)
    
    features['SA MY'] = sade_my.values
    features['SA change'] = sa_1day.values
    
    if plot:
        fig, axs = plt.subplots(2,1,sharex=True)

        axs[0].plot(sa['SADE A MY CalPosSensor'])
        axs[0].plot(sade_my)
        axs[0].set_ylabel('SADE A MY')

        axs[1].plot(sa_1day)
        axs[1].set_ylabel('SADE A MY change')
        axs[1].yaxis.set_label_position('right')
        axs[1].yaxis.tick_right()
        fig.autofmt_xdate()
        fig.savefig('Figures/SA.png')
    
    ###SOLO-HI#####################################################################
        
    print('SOLO-HI')
    print('')
    
    solo_hi_mag = h.load_HK('solo_ANC_sc-lcl-instruments-mag-solohi-b', start, end)
    
    
    solo_hi_mag_day = solo_hi_mag.resample('1D').mean()
    
    features['SoloHI-BDpl'] = solo_hi_mag_day['B_LCL3_10 SoloHI-BDpl TM'].values
    features['SoloHI-BDpl grad'] = np.gradient(features['SoloHI-BDpl'].values)
    
    bins = np.linspace(features['SoloHI-BDpl'].min(),-103,20)
    bins = np.concatenate((bins, np.array([-50, 100])))
    features = h.reduce_cardinality(features, 'SoloHI-BDpl', 0, bins = np.sort(bins))
    
    bins = np.arange(-5,5+0.5,0.5)
    bins = np.concatenate((np.array([-100]), bins, np.array([100])))
    features = h.reduce_cardinality(features, 'SoloHI-BDpl grad', N, bins = np.sort(bins))
    
    
    if args.plot:
        fig, axs = plt.subplots(2,1,sharex=True)
        
        # axs.plot(th_swa_rpw.index , th_swa_rpw['ANP_1_1_7 RPW SCM SRP'])
        
        axs[0].plot(features['Date'], features['SoloHI-BDpl'])
        axs[0].set_ylabel('SoloHI-BDpl')
        
        axs[1].plot(features['Date'], features['SoloHI-BDpl grad'])
        
        fig.savefig("Figures/SoloHI-BDpl.png")


    ######Pointing position of craft###############################################
    
    print('Pointing position')
    print('')

    pointing = h.load_HK('solo_ANC_sc-pointing', start, end)

    pointing_day = pointing.resample('1D').mean()

    features['SUN_X'] = pointing_day['GFE_EST_DIR_SUN_EST-X'].values
    features['SUN_Y'] = pointing_day['GFE_EST_DIR_SUN_EST-Y'].values
    features['SUN_Z'] = pointing_day['GFE_EST_DIR_SUN_EST-Z'].values

    bins = np.linspace(features['SUN_X'].min(),-103,20)
    bins = np.concatenate((bins, np.array([-50, 100])))
    features = h.reduce_cardinality(features, 'SUN_X', 0, bins = np.sort(bins))
    features = h.reduce_cardinality(features, 'SUN_Y', 0, bins = np.sort(bins))
    features = h.reduce_cardinality(features, 'SUN_Z', 0, bins = np.sort(bins))

    
    
    ###############################################################################
    ### HGA #######################################################################
    ###############################################################################
    '''
    print('HGA')
    print('')
    
    if start < datetime(2020,9,1):
        print('No data before 2020-09-03')
    
    # High Gain antenna. I am putting in azimuth and elevation, plus an encoded colum
    # for if there were changes in a day
    
    hga = h.load_HK('solo_ANC_sc-hga-mga', start, end)
    
    hga_day = hga.resample('1D').apply(lambda x: h.mode(x, 20))
    
    def encode(arr):
        # get the unique values 
        
        unique = np.unique(arr)
        
        # just 0s or 1
        if 1 in unique and -1 in unique:
            return 2
        elif 1 in unique:
            return 1
        elif -1 in unique:
            return -1
        else:
            return 0


    # smooth out the data so the diff works better
    hga_az_1min = hga.resample('10T').mean()
    hga_diff_az = np.gradient(hga_az_1min['HGA Acquired Azimuth'])

    threshold = 10

    hga_diff_az_encoded = np.where(hga_diff_az > threshold, 1, hga_diff_az)
    hga_diff_az_encoded = np.where(hga_diff_az < -threshold, -1, hga_diff_az_encoded)
    hga_diff_az_encoded = np.where((hga_diff_az > -threshold) & (hga_diff_az < threshold), 0, hga_diff_az_encoded)

    hga_az_1min['diff'] = hga_diff_az_encoded

    hga_az_1day = hga_az_1min['diff'].resample('1D').apply(encode)
    
    # smooth out the data so the diff works better
    hga_el_1min = hga.resample('10T').mean()
    hga_diff_el = np.gradient(hga_el_1min['HGA Acquired Elevation'])

    threshold = 1

    hga_diff_el_encoded = np.where(hga_diff_el > threshold, 1, hga_diff_el)
    hga_diff_el_encoded = np.where(hga_diff_el < -threshold, -1, hga_diff_el_encoded)
    hga_diff_el_encoded = np.where((hga_diff_el > -threshold) & (hga_diff_el < threshold), 0, hga_diff_el_encoded)

    hga_el_1min['diff'] = hga_diff_el_encoded

    hga_el_1day = hga_el_1min['diff'].resample('1D').apply(encode)

    features['HGA azimuth'] = hga_day['HGA Acquired Azimuth'].values
    features['HGA elevation'] = hga_day['HGA Acquired Elevation'].values
    features['HGA azimuth change'] = hga_az_1day.values
    features['HGA elevation change'] = hga_el_1day.values

    N = 20
    for feature_name in ['HGA azimuth', 'HGA elevation']:
        features = h.reduce_cardinality(features, feature_name, N)

    if args.plot:
        fig, axs = plt.subplots(4,1,sharex=True)

        axs[0].scatter(features['Date'], features['HGA azimuth'])
        axs[0].set_ylabel('Azimuth')
        axs[1].plot(features['Date'], features['HGA azimuth change'])
        axs[1].set_ylabel('Azimuth change')
        axs[1].yaxis.set_label_position('right')
        axs[1].yaxis.tick_right()
        
        axs[2].scatter(features['Date'], features['HGA elevation'])
        axs[2].set_ylabel('Elevation')
        axs[3].plot(features['Date'], features['HGA elevation change'])
        
        axs[3].set_ylabel('Elevation change')
        axs[3].yaxis.set_label_position('right')
        axs[3].yaxis.tick_right()
        fig.autofmt_xdate()
        fig.savefig('Figures/HGA.png')

    
    '''
    ###############################################################################
    ### lag #######################################################################
    ###############################################################################
    # want to include some information about the past few days
    # this is what I was trying to do with days since drop
    # still want to include this parameter since it can get me really far back in the past
    
    # this can also help account for things that are slighly misaligned
    # If I got the day wrong somehow?

    
    

    if args.history > 0:
        # TH MAG
        features = h.add_history(features,'time_since_T_high', n_days_back)
        features = h.add_history(features,'cumulative', n_days_back)
        features = h.add_history(features,'OBS T', n_days_back)
        features = h.add_history(features,'IBS T', n_days_back)
        features = h.add_history(features,'OBS minus IBS T', n_days_back)
        
        # duty cycle
        features = h.add_history(features,'dutycycle', n_days_back)
        
        # RPW?
        features = h.add_history(features,'SCM T', n_days_back)
        features = h.add_history(features,'SCM T grad', n_days_back)
        
        # LCL
        #features = h.add_history(features,'Mag PWR', n_days_back)
        #features = h.add_history(features,'Mag A', n_days_back)
        #features = h.add_history(features,'No time A off', n_days_back)

        # Pointing
        features = h.add_history(features,'SUN_X', n_days_back)
        features = h.add_history(features,'SUN_Y', n_days_back)
        features = h.add_history(features,'SUN_Z', n_days_back)
        
        # # SA
        features = h.add_history(features,'SA MY', n_days_back)
        features = h.add_history(features,'SA change', n_days_back)

        # # SOLOHI
        features = h.add_history(features,'SoloHI-BDpl', n_days_back)
        features = h.add_history(features,'SoloHI-BDpl grad', n_days_back)
        
        # # HGA
      #  features = h.add_history(features,'HGA azimuth', n_days_back)
      #  features = h.add_history(features,'HGA azimuth change', n_days_back)
       # features = h.add_history(features,'HGA elevation', n_days_back)
       # features = h.add_history(features,'HGA elevation change', n_days_back)

    # now trim to the times specified, so I don't have to make up the history
    features = features.loc[features['Date'] >= args.start]
    ###############################################################################
    ### save ######################################################################
    ###############################################################################
    
    # save features to be used by training
    features.to_csv(h.LOCAL_HK_FOLDER + "features.csv", date_format='%Y-%m-%d', index=False)