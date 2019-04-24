from class_metabot import MetaBot
import nibabel as nib
import numpy as np
import tensorflow as tf
import random
from tensorflow import data
from my_utils import *
import matplotlib.pyplot as plt
class DataLoader():
    def __init__(self):
        self.frd = 2
        self.data_limit = 0
        self.data_cnt_tr = 0
        self.data_cnt_tst = 0

        # train data must have both mri and pet images
        self.data_tr = []
        self.data_tr_MRI = []
        self.data_tr_PET = []
        # data with only pet or mri
        self.data_tst = []
        self.data_tst_MRI = []
        self.data_tst_PET = []
        # self.label_tr = []
        # self.label_tst_mri = []
        # self.label_tr_pet = []
        # self.label_tst = [] # we do not need this
        self.meta_info = []
        self.meta_refined = []
        self.meta_tr = []
        self.meta_tst = []

    def make_dataset(self):
        # dataset = data.Dataset.from_tensors(self.data_tr_MRI)
        tr_MRI = tf.convert_to_tensor(self.data_tr_MRI)
        tr_PET = tf.convert_to_tensor(self.data_tr_PET)
        dataset = data.Dataset.from_tensor_slices([self.data_tr_MRI])
        print('dataset count is {}'.format(len(dataset)))
        return dataset

    def get_tr_dataset(self):
        print('get training data for gan')
        print('the training data count is {}'.format(len(self.data_tr)))
        print('data shape is ',np.shape(self.data_tr))
        return self.data_tr

    def get_tr_data(self):
        print('get training data for gan')
        print('the training data count is {}'.format(len(self.data_tr_MRI)))
        print(np.shape(self.data_tr_MRI))
        return self.data_tr_MRI, self.data_tr_PET

    def get_tst_data(self):
        print('get testing data for gan')
        print('the testing data count is {}'.format(len(self.data_tst)))
        print(np.shape(self.data_tst))
        return self.data_tst_MRI, self.data_tst_PET

    def make_data_gan(self):
        # one set is composed of A and B with same person but different domain
        cnt = 0
        for datum in self.meta_info:
            if self.data_limit and self.data_cnt_tr >= self.data_limit:
                break;
            if datum.has_both():
                if not self.is_frd(cnt):
                    self.ld_img_gan_tr(datum)
                    self.data_cnt_tr = self.data_cnt_tr + 1
                else:
                    self.ld_img_gan_tst(datum)
                    self.data_cnt_tst = self.data_cnt_tst + 1
            else: # this datum has mri only image
                pass
            cnt = cnt + 1

    def refine_meta(self):
        # one set is composed of A and B with same person but different domain
        cnt = 0
        for datum in self.meta_info:
            if self.data_limit and self.data_cnt_tr >= self.data_limit:
                break;
            if datum.has_both():
                if not self.is_frd(cnt):
                    self.meta_tr.append(datum)
                    self.data_cnt_tr = self.data_cnt_tr + 1
                else:
                    self.meta_tst.append(datum)
                    self.data_cnt_tst = self.data_cnt_tst + 1
            else: # this datum has mri only image
                self.meta_refined.append(datum)
                pass
            cnt = cnt + 1

    def is_frd(self, index):
        if index%self.frd == 0: return True
        return False

    def ld_img(self, datum, type = 'mri'):
        path = self.get_path(datum, type)
        # print(path)
        # data = transform(nib.load(path).get_data(), npx=256, is_crop=False, resize_w=256)
        data = nib.load(path).get_data()
        # return the shape [256,256,256,1] for concatenating mri and pet images
        return np.expand_dims(data, axis=3)

    def get_path(self, datum, type = 'mri'):
        path = ''
        if type == 'mri': path = datum.get_mri()
        elif type == 'pet': path = datum.get_pet()
        return path

    def get_batch(self, num, type):
        batch_img_l = []
        data_l = self.shuffle(self.meta_tr, num) if type == 'train' else self.shuffle(self.meta_tst, num)
        for datum in data_l:
            batch_img_l.append(self.ld_img_gan_tr_batch(datum))
        return batch_img_l, data_l

    def assert_both(self, datum):
        if datum.has_both(): return True
        else:
            print('A datum is detected which does not have both modality.')
            print(datum.get_mri())
            print(datum.get_pet())

    def reshape(self, img):
        pass

    def ld_img_gan_tr_batch(self, datum):
        self.assert_both(datum)
        mri = self.ld_img(datum,'mri')
        pet = self.ld_img(datum,'pet')
        img = [mri,pet]
        # img = np.reshape(img, [256,256,256,2])

        img = np.concatenate((mri,pet), axis=3)
        # print(img.shape)
        # img_tst = img[:,:,:,1]
        # plt.imshow(img_tst[128])
        # plt.show()
        # assert False # pass!
        return img

    def ld_img_gan_tr(self, datum):
        assert datum.has_both()
        # print('read mri file : {}'.format(mri_path))
        # print('read pet file : {}'.format(pet_path))
        # mri = nib.load(mri_path).get_data()
        mri = self.ld_img(datum, 'mri')
        pet = self.ld_img(datum, 'pet')
        # self.data_tr_MRI.append(mri)
        # self.data_tr_PET.append(pet)
        # data = np.concatenate((mri,pet), axis=3)
        data = [mri,pet] # [2, 256, 256, 256]
        data = np.reshape(data, [256,256,256,2]) # should use concatenate instead of reshape!
        print(np.shape(data))
        # print(data.shape)
        # self.data_tr.append([mri,pet])
        self.data_tr.append(data)

    def ld_img_gan_tst(self, datum):
        assert datum.has_mri()
        mri = self.ld_img(datum, 'mri')
        pet = self.ld_img(datum, 'pet')
        none = [0]
        # self.data_tst_MRI.append(mri)
        # self.data_tst_PET.append(pet)
        # data = [mri,pet]
        # data = np.concatenate((mri,pet), axis=2)
        # tf.reshape(data,[256,256,256,2])  # should use concatenate instead of reshape!
        self.data_tst.append([mri,pet])
        # self.data_tst.append([mri, none])

    def set_meta_info(self, meta_l):
        print('set meta information of {} subjects.'.format(len(meta_l)))
        self.meta_info = meta_l

    def shuffle(self, l, num):
        # print(l)
        shuffled = random.sample(l, num)
        return shuffled

class DataInfo():
    def __init__(self):
        self.meta = []
        self.format = 'hdr'
        # self.format = 'img'
        self.grp = ['AD', 'MCI', 'NORMAL']

    def get_meta_info(self):
        return self.meta

    def rd_fld(self, dir):
        bot = MetaBot('')
        fld_l = bot.get_file_list(dir)
        print('start read directory')
        print('folder count is {}'.format(len(fld_l)))
        for fld in fld_l:
            path = bot.join_path(dir,fld)
            if bot.is_dir(path):
                self.rd_person(path)
        print('reading data info from fld is done')
        print('data count is {}'.format(len(self.meta)))
        del bot

    def rd_person(self, dir):
        bot = MetaBot('')
        file_l = bot.get_file_list(dir)
        # print(file_l)
        datum = Datum()
        tmp_path = []
        for file in file_l:
            file_path = bot.join_path(dir, file)
            # assert bot.is_exist(file_path)
            tmp_path.append(file_path)
        datum.set_info(tmp_path, self.get_gr_num(dir))
        datum.print()
        self.meta.append(datum)
        del bot

    def is_format(self, name):
        if self.format in name: return True
        return False

    def get_gr_num(self, path):
        index = 0
        for gr in self.grp:
            if gr in path:
                return index
            index = index+1
        print('cannot find group number in get_gr_num.')

    def get_meta_info(self):
        return self.meta

class Datum():
    def __init__(self):
        self.name = 'None'
        self.gr_num = 0
        self.mri_is = False
        self.pet_is = False
        self.mri_path = 'None'
        self.pet_path = 'None'
        self.info = []

    def set_info(self, path_l, gr_num):
        # hdr and img file must be together.
        self.gr_num = gr_num
        tmp_mri = []
        tmp_pet = []
        for path in path_l:
            path_split = path.split('/')
            # print(path_split)
            self.name = path_split[-2]
            if self.is_mri(path):
                tmp_mri.append(path)
            if self.is_pet(path):
                tmp_pet.append(path)

        if len(tmp_mri) == 2:
            self.mri_is = True
            self.mri_path = tmp_mri[0]
        if len(tmp_pet) == 2:
            self.pet_is = True
            self.pet_path = tmp_pet[0]
        self.set_print()

    def is_pet(self,name):
        if 'pet' in name:
            return True
        return False

    def is_mri(self, name):
        if 'strip' in name:
            return True
        return False

    def set_print(self):
        self.info.append(self.name)
        self.info.append(self.gr_num)
        self.info.append(self.mri_is)
        self.info.append(self.mri_path)
        self.info.append(self.pet_is)
        self.info.append(self.pet_path)

    def print(self):
        print(self.info)

    def has_both(self):
        if self.has_mri() and self.has_pet(): return True
        return False

    def has_mri(self):
        if self.mri_is: return True
        return False

    def has_pet(self):
        if self.pet_is: return True
        return False

    def get_name(self):
        return self.name

    def get_pet(self):
        return self.pet_path

    def get_mri(self):
        return self.mri_path

    def get_gr(self):
        return self.gr_num

class Loader():
    def __init__(self):
        pass

    def read_data(self):
        bot = MetaBot('')
        base_fld = '/home/sp/Datasets/MRI_PETDATA/MRI_PETDATA_0927'
        class_name = ['linearToJakobAD','linearToJakobMCI','linearToJakobNORMAL']
        meta_loader = DataInfo()
        for name in class_name:
            meta_loader.rd_fld(bot.join_path(base_fld,name))
            info = meta_loader.get_meta_info()
        self.data_loader = DataLoader()
        self.data_loader.set_meta_info(info)
        self.data_loader.refine_meta()
        # self.data_loader.make_data_gan()

    def load_batch(self, num, type = 'train'):
        batch, meta = self.data_loader.get_batch(num, type)
        return batch, meta

    def load_dataset_tr(self):
        return self.data_loader.get_tr_dataset()

    def load_data_tr(self):
        data_tr_MRI, data_tr_PET = self.data_loader.get_tr_data()
        # data_tst = data_loader.get_tst_data()
        # dataset = data_loader.make_dataset()
        return data_tr_MRI, data_tr_PET

    def load_data_tst(self):
        data_tst_MRI, data_tst_PET = self.data_loader.get_tst_data()
        return data_tst_MRI, data_tst_PET

# load_data()
# path = '/home/sp/Datasets/MRI_PETDATA/MRI_PETDATA_0927/linearToJakobNORMAL/137_S_0283/pet-affine.hdr'
# img = nib.load(path).get_data()
# dataset_tst = data.Dataset.from_tensors(tf.random_uniform([4,10,23,120]))
# print(dataset_tst)
