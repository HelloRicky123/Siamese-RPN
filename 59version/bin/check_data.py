import cv2
import os
import xml.etree.ElementTree as ET

from IPython import embed
from glob import glob

if __name__ == '__main__':
    # image_path = '/mnt/usershare/zrq/pytorch/lab/model/zhangruiqi/ytb_vid/benchmark.ytbb/phIXsE50tCY_19'
    image_path = '/mnt/diska1/YT-BB/v2/youtube_dection_frame_temp/phIXsE50tCY_19'
    anno_path = '/mnt/usershare/zrq/pytorch/lab/model/zhangruiqi/ytb_vid/benchmark.ytbb/xml'
    images_path = glob(image_path + '/*')
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame in images_path:
        file_name = frame.split('/')[-1][:15]
        image = cv2.imread(frame)
        frame_anno = anno_path + '/' + file_name + '.xml'
        tree = ET.parse(frame_anno)
        root = tree.getroot()
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            image = cv2.rectangle(image, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
                                  color=(0, 255, 255))
            image = cv2.putText(image, str(trkid), (int(bbox[0]), int(bbox[1])), font, 1, (0, 255, 255))
        cv2.imwrite(file_name + '.jpg', image)
