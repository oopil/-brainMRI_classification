import os
import scipy.misc
import tensorflow as tf

def save_images_from_event(fn, tag, output_dir='./board_image'):
    print('save images from event')
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            # print(e)
            for v in e.summary.value:
                print(v.tag)
                # if v.tag == tag:
                #     im = im_tf.eval({image_str: v.image.encoded_image_string})
                #     output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                #     print("Saving '{}'".format(output_fn))
                #     scipy.misc.imsave(output_fn, im)
                #     count += 1
                # count += 1
        print('total count : ', count)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    event_file = './log/train/201962772234/events.out.tfevents.1561620157.dgist'
    save_images_from_event(event_file, 'Model/patch0/attent1/attent1_decode/attention_mask/image/0')
