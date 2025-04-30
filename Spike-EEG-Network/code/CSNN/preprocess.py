import os
import gc
import psutil
import torch
import datetime
import scipy.io
import numpy as np
import scipy.stats
from scipy import signal
from scipy.signal import welch
from scipy.signal import butter, lfilter
from torch.utils.data import DataLoader, TensorDataset, Dataset


class DataLoadEEG:
    def __init__(self, subject='all', band=[0.3, 45], fs_orig=500, fs_target=100,
                 parent_directory='/data/dataset/EAV_complete/EAV/'):
        self.subject = subject
        self.band = band
        self.parent_directory = parent_directory
        self.fs_orig = fs_orig
        self.fs_target = fs_target
        self.seg = []
        self.label = []
        self.seg_f_div = []
        self.label_div = []

        self.freq_bands = {
            'delta': (0.5, 4),  
            'theta': (4, 8),
            'alpha_low': (8, 10), 
            'alpha_high': (10, 13),
            'beta_low': (13, 20), 
            'beta_high': (20, 30),
            'gamma_low': (30, 40), 
            'gamma_high': (40, 45)
        }

        self.brain_regions = {
            'frontal': [0, 1, 2, 3, 4, 5, 6, 7],      # prefrontal
            'central': [8, 9, 10, 11, 12, 13],        # centre
            'temporal': [14, 15, 16, 17, 18, 19],     # temporal
            'parietal': [20, 21, 22, 23],             # parietal
            'occipital': [24, 25, 26, 27, 28, 29]     # occipital
        }

    def data_mat(self):
        subject = f'subject{self.subject:02d}'
        eeg_folder = os.path.join(self.parent_directory, subject, 'EEG')
        eeg_file_path = os.path.join(eeg_folder, f'{subject}_eeg.mat')
        label_file_path = os.path.join(eeg_folder, f'{subject}_eeg_label.mat')

        if not os.path.exists(eeg_file_path):
            raise FileNotFoundError(f'EEG data not found for {subject} at {eeg_file_path}')

        try:
            mat = scipy.io.loadmat(eeg_file_path)
            cnt_ = np.array(mat.get('seg1') if 'seg1' in mat else mat.get('seg'))
            
            if not os.path.exists(label_file_path):
                raise FileNotFoundError(f'Label data not found for {subject}')
            mat_y = scipy.io.loadmat(label_file_path)
            label = np.array(mat_y.get('label'))

            self.seg = np.transpose(cnt_, [1, 0, 2])
            self.label = label

            print(f'\nSuccessfully loaded EEG data for {subject}')
            print(f'EEG data shape: {self.seg.shape}')
            print(f'Label shape: {self.label.shape}')
            return True
            
        except Exception as e:
            print(f'Error loading data for {subject}: {str(e)}')
            return False

    def downsampling(self):
        ch, t, tri = self.seg.shape
        factor = self.fs_target / self.fs_orig
        tm = np.reshape(self.seg, [ch, t * tri], order='F')
        tm2 = signal.resample_poly(tm, up=1, down=int(self.fs_orig / self.fs_target), axis=1)
        self.seg = np.reshape(tm2, [ch, int(t * factor), tri], order='F')

    def bandpass(self):
        ch, t, tri = self.seg.shape
        dat = np.reshape(self.seg, [ch, t * tri], order='F')
        sos = butter(5, self.band, btype='bandpass', fs=self.fs_target, output='sos')
        fdat = []
        for i in range(np.size(dat, 0)):
            tm = signal.sosfilt(sos, dat[i, :])
            fdat.append(tm)
        self.seg_f = np.array(fdat).reshape([ch, t, tri], order='F')

    def compute_psd(self, eeg_data):
        """
        Computing PSD features for EEG data
        Args:
            eeg_data: [channels, samples]
        Returns:
            enhanced_psd_features: [8, channels]
        """
        # computing basic PSD features
        psd_features = []
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            freqs, psd = signal.welch(eeg_data,
                                      fs=self.fs_target,
                                      nperseg=256,
                                      noverlap=128,
                                      window='hann')
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(psd[:, freq_mask], axis=1)
            
            # enhance the power of certain bands
            if 'beta' in band_name:
                band_power *= 1.3
            elif 'gamma' in band_name:
                band_power *= 1.5
            elif 'alpha' in band_name:
                band_power *= 1.2
                
            psd_features.append(band_power)
            
        return np.array(psd_features)  # shape [8, channels]

    def compute_time_domain_features(self, eeg_data):
        """
        compute time domain features
        """
        features = []
        # mean
        mean = np.mean(eeg_data, axis=1)
        features.append(mean)
        # std
        std = np.std(eeg_data, axis=1)
        features.append(std)
        # kurtosis
        kurtosis = scipy.stats.kurtosis(eeg_data, axis=1)
        features.append(kurtosis)
        # skewness
        skewness = scipy.stats.skew(eeg_data, axis=1)
        features.append(skewness)
        # zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(eeg_data), axis=1), axis=1)
        features.append(zero_crossings / eeg_data.shape[1])
        # activity (Hjorth params-activity)
        activity = np.var(eeg_data, axis=1)
        features.append(activity)
        # mobility (Hjorth params-mobility)
        diff1 = np.diff(eeg_data, axis=1)
        mobility = np.sqrt(np.var(diff1, axis=1) / np.var(eeg_data, axis=1))
        features.append(mobility)
        
        return np.array(features)  # shape [7, channels]
    
    def compute_connectivity_features(self, eeg_data):
        """
        compute connectivity features
        """
        n_channels = eeg_data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        # corr between channels
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                corr = np.corrcoef(eeg_data[i], eeg_data[j])[0, 1]
                connectivity_matrix[i, j] = connectivity_matrix[j, i] = corr
                
        region_connectivity = []
        
        # average connectivity within brain regions
        for region, channels in self.brain_regions.items():
            if len(channels) > 1:  
                region_conn = []
                for i in range(len(channels)):
                    for j in range(i+1, len(channels)):
                        ch_i, ch_j = channels[i], channels[j]
                        region_conn.append(connectivity_matrix[ch_i, ch_j])
                if region_conn:
                    region_connectivity.append(np.mean(region_conn))
        
        # interneuronal connectivity
        inter_region_conn = []
        regions = list(self.brain_regions.keys())
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                region_i_channels = self.brain_regions[regions[i]]
                region_j_channels = self.brain_regions[regions[j]]
                conn_values = []
                for ch_i in region_i_channels:
                    for ch_j in region_j_channels:
                        conn_values.append(connectivity_matrix[ch_i, ch_j])
                if conn_values:
                    inter_region_conn.append(np.mean(conn_values))
                    
        all_connectivity = np.array(region_connectivity + inter_region_conn)
        
        padded_connectivity = np.zeros(n_channels)
        padded_connectivity[:min(len(all_connectivity), n_channels)] = all_connectivity[:min(len(all_connectivity), n_channels)]
        
        return padded_connectivity.reshape(1, -1)  # shape [1, channels]
    
    def data_div(self):
        ch, t, tri = self.seg.shape
        print(f"shape of origin data: {self.seg_f.shape}, total size: {self.seg_f.size}")

        # tm1 = self.seg_f.reshape((30, 500, 4, 200), order='F')
        total_elements = self.seg_f.size
        if total_elements % (30 * 500 * 4) == 0:
            samples_per_class = total_elements // (30 * 500 * 4)
            tm1 = self.seg_f.reshape((30, 500, 4, samples_per_class), order='F')
        else:
            ch_samples = total_elements // (ch * t)
            tm1 = self.seg_f.reshape((ch, t, ch_samples), order='F')

        self.seg_f_div = tm1.reshape((30, 500, 4 * 200), order='F')
        self.label_div = np.repeat(self.label, repeats=4, axis=1)


        selected_classes = [1, 3, 5, 7, 9]
        label = self.label_div[selected_classes, :]
        selected_indices = np.isin(np.argmax(self.label_div, axis=0), selected_classes)
        label = label[:, selected_indices]
        x = self.seg_f_div[:, :, selected_indices]

        self.seg_f_div = np.transpose(x, (2, 0, 1))
        self.label_div = np.argmax(label, axis=0)

    def data_prepare(self):
        self.data_mat()
        self.downsampling()
        self.bandpass()
        self.data_div()

        samples = self.seg_f_div.shape[0]  # [samples, channels, times]
        all_psd_features = []

        """
        for i in range(samples):
            eeg_sample = self.seg_f_div[i]  # [channels, times]
            # PSD
            psd_features = self.compute_psd(eeg_sample)  # [8, channels]
            # time domain features
            time_features = self.compute_time_domain_features(eeg_sample)  # [7, channels]
            # connectivity features
            connectivity_features = self.compute_connectivity_features(eeg_sample)  # [1, channels]
            
            all_features = np.concatenate([psd_features, time_features, connectivity_features], axis=0)  # [16, channels]
            
            all_psd_features.append(all_features)

        all_psd_features = np.array(all_psd_features)  # [samples, 16, channels]
        print('EEG preporcess completed')
        print(f'enhanced psd features shape: {all_psd_features.shape}')

        return all_psd_features, self.label_div
        """
        
        for i in range(samples):
            eeg_sample = self.seg_f_div[i]  # [channels, times]
            psd = self.compute_psd(eeg_sample)  # [8, channels]
            all_psd_features.append(psd)

        all_psd_features = np.array(all_psd_features)  # [samples, 8, channels]

        print('EEG preprocess completed')
        print(f'PSD features shape: {all_psd_features.shape}')

        return all_psd_features, self.label_div
