from class_metabot import *
# class to handle chosun univ. alzheimer's disease brain MRI data set
class ADBrainMRI(MetaBot):
    def MRI_chosun(self, class_name) -> list:
        '''
        find all subject name, input file path
        :param class_name:
        :return: [folder_path, subj_name, input] list
        '''
        class_path = self.join_path(self.base_path, class_name)
        class_path = self.join_path(class_path, 'T1sag')
        subj_name_list = self.get_file_list(class_path)
        # print(subj_name_list)
        meta_list = []
        for subj in sorted(subj_name_list):
            subj_path = self.join_path(class_path, subj)
            if not self.is_dir(subj_path):
                continue
            T1_path = self.join_path(subj_path, 'T1.nii.gz')
            if not self.is_exist(T1_path):
                print('T1 image does not exists.', T1_path, subj)
                assert False
            folder_path = class_path
            subj_name = subj
            input_path = T1_path

            new_line = [folder_path, subj_name, input_path]
            meta_list.append(new_line)
        # print(new_line)
        return meta_list

    def MRI_chosun_patch_cnn(self, class_name) -> list:
        '''
        find all subject name, input file path
        :param class_name:
        :return: [folder_path, subj_name, input, seg_input] list
        '''
        class_path = self.join_path(self.base_path, class_name)
        subj_name_list = self.get_file_list(class_path)
        # print(subj_name_list)
        meta_list = []
        excluded_list = []
        for subj in sorted(subj_name_list):
            subj_path = self.join_path(class_path, subj)
            subj_path = self.join_path(subj_path, 'processed')
            if not self.is_dir(subj_path):
                continue
            T1_path = self.join_path(subj_path, 'brain.nii')
            label_path = self.join_path(subj_path, 'aparc.DKTatlas+aseg.nii')
            # print(T1_path)
            # print(label_path)
            if not self.is_exist(T1_path):
                print('T1 image does not exists.', T1_path, subj)
                assert False
            if not self.is_exist(label_path):
                print('Label does not exists.', T1_path, subj)
                print('Skip this Subject.')
                excluded_list.append([class_name, subj])
                continue
            folder_path = class_path
            subj_name = subj
            input_path = T1_path + ','+label_path

            new_line = [folder_path, subj_name, input_path]
            meta_list.append(new_line)
            # print(new_line)
        print('Subject which is not included : {}'.format(excluded_list))
        # assert False
        return meta_list
#%%
    def find_all_folder_chosun_AD(self, path, depth):
        file_list = self.get_file_list(path)
        for file_name in file_list:
            new_path = self.join_path(path, file_name)
            if self.is_dir(new_path):
                self.fld_list.append(new_path)
                if depth:
                    self.find_all_folder_chosun_AD(new_path, depth-1)

    def copy_directory_only(self, base_folder_path):
        self.base_folder_path = base_folder_path
        self.find_all_folder_chosun_AD(self.base_folder_path, 2)
        dir_path_list = self.get_fld_list()
        # print("/".join(folder_path_split))
        # print(folder_path_split)
        # print(self.base_folder_path)

        self.copy_folder_path = self.convert_path(self.base_folder_path, 'empty_copy')
        print('copy directory path : ',self.copy_folder_path)
        self.make_directory_sub(self.copy_folder_path)

        for dir_path in dir_path_list:
            # print(self.base_folder_path)
            # print(dir_path)
            # print(self.convert_path(dir_path, 'empty_copy'))
            new_dir_path = self.convert_path(dir_path, 'empty_copy')
            self.make_directory_sub(new_dir_path)
            print(dir_path)
            print(new_dir_path)
            #/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0/mAD/T1sag
            #/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy/mAD/T1sag

            path_list = dir_path.split('/')
            if len(path_list[-1]) > 5:
                processed_dir_path = self.join_path(new_dir_path, 'processed')
                self.make_directory_sub(processed_dir_path)
                print(processed_dir_path)
        return self.copy_folder_path

    def convert_path(self, path, new_dir_name):
        folder_path_split = [e for e in self.base_folder_path.split('/') if e != '']
        path_split        = [e for e in path.split('/') if e != '']
        i=0
        for i, e in enumerate(folder_path_split):
            # print(i,e, path_split[i])
            assert e == path_split[i]

        folder_path_split[-1] = folder_path_split[-1] + '_' + new_dir_name
        folder_path_split = folder_path_split + path_split[i+1:]
        try:
            folder_path_split.remove('T1sag')
        except ValueError:
            pass
        # print('remove t1sag ',self.remove_t1sag(folder_path_split))
        new_dir_path = "/" + "/".join(folder_path_split)
        # print(folder_path_split)
        # print(path_split[i+1:])
        # print(new_dir_path)
        # assert False
        return new_dir_path

    def remove_t1sag(self, path_split):
        if 'T1sag' in path_split:
            return path_split.remove('T1sag')
        return path_split

    def copy_only_useful_file(self, copy_dir_path, subj_list:list, file_list:list)->None:
        target_dir = 'freesurfer/mri/nifti'
        target_copy_dir = 'processed'
        for subj in subj_list:
            folder_path, subj_name, _ = subj
            subj_dir = os.path.join(folder_path, subj_name, target_dir)
            subj_copy_dir = os.path.join(copy_dir_path, subj_name, target_copy_dir)
            for file in file_list:
                file_path = os.path.join(subj_dir, file)
                copy_file_path = os.path.join(subj_copy_dir, file)
                print(file_path)
                print(copy_file_path)
                self.copy_file(file_path, copy_file_path)

        pass

    def extract_score(self):
        '''
        extrack score from segmentation and parcellation label - volumn and thickness...
        and make excel file
        :return:
        '''
        pass