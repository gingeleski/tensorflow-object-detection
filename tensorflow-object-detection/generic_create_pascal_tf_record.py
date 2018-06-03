# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./generic_create_pascal_tf_record --data_dir=/home/user/VOCdevkit/2007
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Path to PASCAL VOC dataset root where Annotations, ImageSets, JPEGImages are.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '.', '(Relative) path to output directory for TFRecord file.')
flags.DEFINE_string('label_map_path', './pascal_label_map.pbtxt',
                    '(Relative) path to label map proto.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances.')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_record(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """
  Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid image file
  """
  full_path_root = os.path.join(dataset_directory, image_subdirectory, data['filename'])
  full_path = full_path_root + '.JPG'
  is_png = False
  # If not JPG try JPEG then PNG
  if os.path.isfile(full_path) == False:
    full_path = full_path_root + '.JPEG'
    if os.path.isfile(full_path) == False:
      full_path = full_path_root + '.PNG'
      is_png = True
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_img = fid.read()
  encoded_img_io = io.BytesIO(encoded_img)
  image = PIL.Image.open(encoded_img_io)
  if image.format != 'JPEG' and image.format != 'PNG':
    raise ValueError('Image format not JPEG or PNG')
  key = hashlib.sha256(encoded_img).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('Set arg must be in {}'.format(SETS))
  set_name = FLAGS.set

  data_dir = FLAGS.data_dir
  # Validate data directory
  if not data_dir:
    raise ValueError('Did not specify required arg "data_dir"')
  elif os.path.isfile(data_dir) == False:
    raise ValueError('data_dir arg does not appear to be real directory, please check it.')

  output_path_root = FLAGS.output_path
  # Infer the full output path unless it looks like a full path
  if output_path_root.startswith('/') == False:
    # Then it's a relative path like we were aiming for
    output_path_root = os.path.join(data_dir, output_path_root)
  if os.path.isfile(output_path_root) == False:
    raise ValueError('Could not validate that target output path exists, please check arg.')
  tf_filename = set_name + '.tfrecord'
  output_path = os.path.join(output_path_root, tf_filename)
  writer = tf.python_io.TFRecordWriter(output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  samples_path_root = os.path.join(data_dir, 'ImageSets', 'Main')
  if os.path.isfile(samples_path_root) == False:
    raise ValueError('Expected ImagePath/Main to be in samples path, did not find.')
  # Get the names of .txt files in there and infer the one we need
  for rootpath, dirnames, filenames in os.walk(samples_path_root):
    for filename in filenames:
      if filename.endswith('.txt'):
        # In worst case we'll try whatever ends in TXT there
        samples_path = filename
        # But preferably the filename has the set name in it
        if filename.contains(set_name):
          break

  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
  # Should not need to validate samples_list since we derived it from existing TXT's
  samples_list = dataset_util.read_samples_list(samples_path)
  for idx, sample in enumerate(samples_list):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(samples_list))
    path = os.path.join(annotations_dir, sample + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_record = dict_to_tf_record(data, FLAGS.data_dir, label_map_dict,
                                    FLAGS.ignore_difficult_instances)
    writer.write(tf_record.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
