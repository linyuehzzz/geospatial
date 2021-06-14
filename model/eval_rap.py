import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np


def parse_rec_region(filename, region):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)

  w = int(float(tree.find("size").find("width").text))
  h = int(float(tree.find("size").find("height").text))

  objects = []
  for obj in tree.findall("object"):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find("bndbox")
    obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
	                      int(float(bbox.find('ymin').text)),
	                      int(float(bbox.find('xmax').text)),
	                      int(float(bbox.find('ymax').text))]
              
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
      objects.append(obj_struct)

  return objects



def voc_ap_region(rec, prec, use_07_metric=True):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap



def voc_eval_region(detpath, annopath, imagenames, classname, cachedir, region, ovthresh=0.5, use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname,
                              cachedir, roibbox, [ovthresh], [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
      detpath.format(imagename) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagenames: Names for the list of images.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations and detections in a pickle file.
  region: Region of interest in image where mAP will be calculated.
      ([xmin, ymin, xmax, ymax]) 
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation.
      (default False)
  """

  # create cache dir
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)

  # first load annots
  cachefile_annots = os.path.join(cachedir, '%s_annots.pkl' % classname)
  # load annotations
  recs_annots = {}
  for i, imagename in enumerate(imagenames):
    recs_annots[imagename] = parse_rec_region(os.path.join(annopath, imagename + ".xml"), region)
  with open(cachefile_annots, 'wb') as f:
    pickle.dump(recs_annots, f)

  # extract objects for this class
  class_recs_annots = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs_annots[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs_annots[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

  # read dets
  cachefile_dets = os.path.join(cachedir, '%s_dets.pkl' % classname)
  # load detections
  recs_dets = {}
  for i, imagename in enumerate(imagenames):
    recs_dets[imagename] = parse_rec_region(os.path.join(detpath, imagename + ".xml"), region)
  with open(cachefile_dets, 'wb') as f:
    pickle.dump(recs_dets, f)

  # extract objects for this class
  class_recs_dets = {}
  image_ids = []
  BB = []
  for imagename in imagenames:
    for obj in recs_dets[imagename]:
      if obj['name'] == classname:
        image_ids.append(imagename)
        BB.append(obj['bbox'])
  image_ids = np.array(image_ids)
  BB = np.array(BB)

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  # go down dets and mark TPs and FPs
  for d in range(nd):
    R = class_recs_annots[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)

    if BBGT.size > 0:
      # compute overlaps
      # intersection
      ixmin = np.maximum(BBGT[:, 0], bb[0])
      iymin = np.maximum(BBGT[:, 1], bb[1])
      ixmax = np.minimum(BBGT[:, 2], bb[2])
      iymax = np.minimum(BBGT[:, 3], bb[3])
      iw = np.maximum(ixmax - ixmin + 1., 0.)
      ih = np.maximum(iymax - iymin + 1., 0.)
      inters = iw * ih

      # union
      uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
              (BBGT[:, 2] - BBGT[:, 0] + 1.) *
              (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
      
      overlaps = inters / uni
      ovmax = np.max(overlaps)
      jmax = np.argmax(overlaps)

    if ovmax > ovthresh:
      if not R['difficult'][jmax]:
        if not R['det'][jmax]:
          tp[d] = 1.
          R['det'][jmax] = 1
        else:
          fp[d] = 1.
    else:
      fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap_region(rec, prec, use_07_metric)
  if len(fp) != 0 and len(tp) != 0:
    acc = tp[-1] / (float(npos) + fp[-1])
    rec_out = rec[-1]
    prec_out = prec[-1]
  else:
    acc = rec_out = prec_out = 0.

  return rec_out, prec_out, ap, acc