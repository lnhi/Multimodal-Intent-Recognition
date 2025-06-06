import logging
import os
import numpy as np
import pickle
# from imblearn.over_sampling import SMOTE

__all__ = ['AudioDataset']

class AudioDataset:

    def __init__(self, args, base_attrs):
        
        self.logger = logging.getLogger(args.logger_name)
        audio_feats_path = os.path.join(base_attrs['data_path'], args.audio_data_path, args.audio_feats_path)
        
        if not os.path.exists(audio_feats_path):
            raise Exception('Error: The directory of audio features is empty.')
        
        self.feats = self.__load_feats(audio_feats_path, base_attrs)

        self.feats = self.__padding_feats(args, base_attrs)

    def __load_feats(self, audio_feats_path, base_attrs):

        self.logger.info('Load Audio Features Begin...')
        with open(audio_feats_path, 'rb') as f:
            audio_feats = pickle.load(f)
        
        train_feats = [audio_feats[x] for x in base_attrs['train_data_index']]
        dev_feats = [audio_feats[x] for x in base_attrs['dev_data_index']]
        test_feats = [audio_feats[x] for x in base_attrs['test_data_index']]   

        # if 'train_labels' not in base_attrs:
        #     raise KeyError("Key 'train_labels' is missing in base_attrs. Ensure train_labels is provided.")
    
        # train_labels = base_attrs['train_labels']

        # # Chuyển train_feats và train_labels sang dạng numpy array
        # X_train = np.array(train_feats)  # Đặc trưng âm thanh
        # y_train = np.array(train_labels, dtype=int)  # Nhãn

        # # Áp dụng SMOTE để cân bằng dữ liệu
        # smote = SMOTE(random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # # Log lại phân phối nhãn sau SMOTE
        # from collections import Counter
        # self.logger.info(f"Label distribution after SMOTE: {Counter(y_resampled)}")

        # # Chuyển X_resampled về danh sách để giữ định dạng ban đầu
        # train_feats_resampled = X_resampled.tolist() 

        self.logger.info('Load Audio Features Finished...')
        
        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }


    def __padding(self, feat, audio_max_length, padding_mode = 'zero', padding_loc = 'end'):
        """
        padding_mode: 'zero' or 'normal'
        padding_loc: 'start' or 'end'
        """
        assert padding_mode in ['zero', 'normal']
        assert padding_loc in ['start', 'end']

        audio_length = feat.shape[0]
        if audio_length >= audio_max_length:
            return feat[audio_max_length, :]

        if padding_mode == 'zero':
            pad = np.zeros([audio_max_length - audio_length, feat.shape[-1]])
        elif padding_mode == 'normal':
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (audio_max_length - audio_length, feat.shape[1]))
        
        if padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis = 0)
        else:
            feat = np.concatenate((feat, pad), axis = 0)

        return feat

    def __padding_feats(self, args, base_attrs):

        audio_max_length = base_attrs['benchmarks']['max_seq_lengths']['audio']

        padding_feats = {}

        for dataset_type in self.feats.keys():
            feats = self.feats[dataset_type]

            tmp_list = []

            for feat in feats:
                feat = np.array(feat)
                padding_feat = self.__padding(feat, audio_max_length, padding_mode=args.padding_mode, padding_loc=args.padding_loc)
                tmp_list.append(padding_feat)

            padding_feats[dataset_type] = tmp_list

        return padding_feats   