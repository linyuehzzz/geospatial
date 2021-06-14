import os
import numpy as np
import xml.etree.ElementTree as ET

def parse_rec_cnt(filename, region):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)

  w = int(float(tree.find("size").find("width").text))
  h = int(float(tree.find("size").find("height").text))

  objects = []
  for obj in tree.findall("object"):
    bbox = obj.find("bndbox")
    xmin = int(float(bbox.find("xmin").text))
    ymin = int(float(bbox.find("ymin").text))
    xmax = int(float(bbox.find("xmax").text))
    ymax = int(float(bbox.find("ymax").text))
    xmid = (xmin + xmax) / 2.
    ymid = (ymin + ymax) / 2.

    dw = w / region.shape[1]
    dh = h / region.shape[0]

    x = int(xmid // dw)
    y = int(ymid // dh)

    if region[y, x] != 0:
      objects.append([xmin, ymin, xmax, ymax])

  return objects


def count(imagename, annopath, detpath, region, ovthresh=0.5):
  recs_annots = parse_rec_cnt(os.path.join(annopath, imagename + ".xml"), region)
  recs_dets = parse_rec_cnt(os.path.join(detpath, imagename + ".xml"), region)

  if len(recs_annots) == 0 or len(recs_dets) == 0:
    return 0, 0, 0
  
  det_cnt = 0
  for det in recs_dets:
    for anno in recs_annots:
      # Intersection
      ixmin = max(anno[0], det[0])
      iymin = max(anno[1], det[1])
      ixmax = min(anno[2], det[2])
      iymax = min(anno[3], det[3])
      iw = ixmax - ixmin + 1.
      ih = iymax - iymin + 1.
      inters = iw * ih
      
      # Union
      uni = (anno[2] - anno[0] + 1.) * (anno[3] - anno[1] + 1.) 
      + (det[2] - det[0] + 1.) * (det[3] - det[1] + 1.) - inters
      
      if inters/uni >= ovthresh:
        det_cnt += 1

  return len(recs_annots), det_cnt