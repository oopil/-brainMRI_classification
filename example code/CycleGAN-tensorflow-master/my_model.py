from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import nibabel
from dataloader import Loader
from collections import namedtuple
from my_module import *
from my_utils import *

class cyclegan(object):
    def __init__(self, sess):
        self.sess = sess

    def read_data(self):
        self.loader = Loader()
        self.loader.read_data()

    def load_batch(self, num, type):
        return self.loader.load_batch(num, type)

    def load_data_tr(self):
        return self.loader.load_dataset_tr()

    def load_data_tst(self):
        return self.loader.load_data_tst()

    def set_hyperparam(self, args):
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        if args.discrim_model == 'disc':
            self.discriminator = disc
        elif args.discrim_model == 'disc_my':
            self.discriminator = disc_my
        # if args.use_resnet:
        #     self.generator = generator_resnet
        if args.gen_model == 'res':
            self.generator = generator_res
        elif args.gen_model == 'res_2':
            self.generator = generator_res_2
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size, args.ngf, args.ndf, args.output_nc, args.phase == 'train'))
        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_size, self.input_c_dim+self.input_c_dim], name='real_A_and_B_images')
        print('real data shape : ',self.real_data.shape)
        self.real_A = self.real_data[:, :, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        print('real data A and B shape : ',self.real_A.shape, self.real_B.shape)

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        print('fake data A and B shape : ',self.fake_A.shape, self.fake_B.shape)
        print('fake data A_ and B_ shape : ',self.fake_A_.shape, self.fake_B_.shape)

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        print('g loss is builded')
        # why sample is bellow ???
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")

        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")
        print('discriminating sample is builded')
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_size, self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_size, self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # dataset = self.load_data_tr()
        for epoch in range(args.epoch):
            # dataA, dataB = self.load_data_tr()
            # dataset = [dataA, dataB]
            # np.random.shuffle(dataset)
            # batch_idxs = min(len(dataset), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            batch, meta = self.load_batch(self.batch_size, type = 'train')
            batch_idxs = len(batch)

            for idx in range(0, batch_idxs):
                # batch_images = dataset[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = batch
                # batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                #                        dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                # batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                # batch_images = np.array(batch_images).astype(np.float32)
                # batch_images = batch_files[batch_idxs]
                # batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str, g_loss = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum, self.g_loss],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str, d_loss = self.sess.run(
                    [self.d_optim, self.d_sum, self.d_loss],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,

                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f | g loss : %8d | d loss : %4d"  % (
                    epoch, idx, batch_idxs, time.time() - start_time, g_loss, d_loss))

                if np.mod(counter, args.print_freq) == args.print_freq - 1:
                    print('save sample model at {}'.format((counter)))
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == args.save_freq - 1:
                    print('save in checkpoint directory at {}'.format(counter))
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        batch, meta = self.load_batch(self.batch_size, type='train')
        sample_image = batch[0]
        # batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        # sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        # sample_images = np.array(sample_images).astype(np.float32)
        # sample_image = np.reshape(sample_image, [1,256,256,256,2])
        sample_image = np.expand_dims(sample_image, axis=0) #[1,256,256,256,2]
        print('sample image shape: {}'.format(sample_image.shape))
        fake_A, fake_B = self.sess.run( [self.fake_A, self.fake_B], feed_dict={self.real_data: sample_image} )
        real_A = sample_image[:, :, :, :, :self.input_c_dim]
        real_B = sample_image[:, :, :, :, self.input_c_dim:self.input_c_dim+self.input_c_dim]
        print('real image shape: {}'.format(real_A.shape))
        print('fake image shape: {}'.format(fake_A.shape))
        save_images(real_A, './{}/A_real{:02d}_{:04d}'.format(sample_dir, epoch, idx))
        save_images(fake_A, './{}/A_fake{:02d}_{:04d}'.format(sample_dir, epoch, idx))
        save_images(fake_B, './{}/B_fake{:02d}_{:04d}'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        batch, meta = self.load_batch(self.batch_size, type = 'test')
        batch_idxs = len(batch)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")
        for idx in range(batch_idxs):
            sample_file = meta[idx].get_name()
            if args.which_direction == 'MtoP':
                out_var, in_var = (self.testB, self.test_A)
                sample_image = batch[idx][:,:,:,:self.input_c_dim]
            else:
                out_var, in_var = (self.testA, self.test_B)
                sample_image = batch[idx][:,:,:,:self.input_c_dim:self.input_c_dim+self.input_c_dim]
            sample_image = np.reshape(sample_image, [1,256,256,256,1])
            print('Processing image: ' + sample_file)
            # sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
