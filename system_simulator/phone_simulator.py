from system_simulator.base import BasicStateUpdater
import os
import zipfile
import pandas as pd
import config as cfg
"""
************************* Availability ***************************
We construct the client availability according to the public real-world dataset "Users Active 
Time Prediction". This is data dealing with mobile app usage of the customers, where an app 
has some personal information and online active timing of the customers. Whenever the 
customers login in-app and view anything, the app server gets pings from their mobile phone 
indicating that they are using the app.  More information about this dateset is in
https://www.kaggle.com/datasets/bhuvanchennoju/mobile-usage-time-prediction .
Before using this simulator, you should manually download and unzip the original file in kaggle to
ensure the  existence of file `benchmark/RAW_DATA/USER_ACTIVE_TIME/pings.csv`.
"""
class StateUpdater(BasicStateUpdater):
    def __init__(self, objects = [], option = None):
        super().__init__(objects)
        self.option = option
        self.num_clients = len(self.clients)
        self.customer_map = {}
        # availability
        self.init_availability(os.path.join('benchmark', 'RAW_DATA', 'USER_ACTIVE_TIME'))

    def init_availability(self, rawdata_path):
        zipfile_path = os.path.join(rawdata_path,'archive.zip')
        data_path = os.path.join(rawdata_path,'pings.csv')
        if not os.path.exists(data_path) and os.path.exists(zipfile_path):
            f = zipfile.ZipFile(zipfile_path, 'r')
            for file in f.namelist():
                f.extract(file, rawdata_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError("Please download the original dataset in https://www.kaggle.com/datasets/bhuvanchennoju/mobile-usage-time-prediction , and move it into `benchmark/RAW_DATA/USER_ACTIVE_TIME/pings.csv`")
        customers_info = pd.read_csv(os.path.join(rawdata_path, "customers.csv"))
        customers_availability = pd.read_csv(os.path.join(rawdata_path, 'pings.csv'))
        customers_availability['timestamp'] = customers_availability['timestamp'] - customers_availability['timestamp'][0]
        customers_availability_by_time = customers_availability.groupby('timestamp')['id'].apply(list)
        # customers_availability_by_time = customers_availability_by_time.to_frame(name='id')
        # customers_availability_by_time = customers_availability_by_time.reset_index()
        customers_availability_by_id = customers_availability.groupby('id')['timestamp'].apply(list)
        customers_availability_by_id = customers_availability_by_id.to_frame(name='timestamps')
        customers_availability_by_id = customers_availability_by_id.reset_index()
        if self.option['availability'] == 'IDL' or (self.option['availability']=='HIGH' and self.num_clients<len(customers_availability_by_id)) :
            customers_availability_by_id['num_stamps'] = customers_availability_by_id.apply(lambda x: len(x['timestamps']), axis=1)
            customers_availability_by_id = customers_availability_by_id.sort_values(by='num_stamps', ascending=False)
            customers_availability_by_id = customers_availability_by_id['id'].to_list()[:self.num_clients]
            self.customer_map = {cid: customers_availability_by_id[cid] for cid in range(self.num_clients)}
        elif self.option['availability'] == 'LOW' and self.num_clients<len(customers_availability_by_id):
            customers_availability_by_id['num_stamps'] = customers_availability_by_id.apply(lambda x: len(x['timestamps']), axis=1)
            customers_availability_by_id = customers_availability_by_id.sort_values(by='num_stamps', ascending=True)
            customers = customers_availability_by_id['id'].to_list()[:self.num_clients]
            self.customer_map = {cid: customers[cid] for cid in range(self.num_clients)}
        elif self.option['availability']=='RANDOM':
            replacement = True  if self.num_clients>len(customers_availability_by_id) else False
            customers = customers_availability_by_id.sample(n=self.num_clients, replace=replacement)['id'].to_list()
            self.customer_map = {cid: customers[cid] for cid in range(self.num_clients)}
        else:
            raise NotImplementedError("Availability {} has been not implemented.".format(self.option['availability']))
        self.availability_table = customers_availability_by_time
        return

    def update_client_availability(self):
        t = cfg.clock.current_time
        t = t%self.availability_table.index[-1]
        aid = t-t%15
        available_customers = self.availability_table[aid]
        pa, pua = [], []
        for cid in self.all_clients:
            pai = 1.0 if self.customer_map[cid] in available_customers else 0.0
            pa.append(pai)
            pua.append(1.0-pai)
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)