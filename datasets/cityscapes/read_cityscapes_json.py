#!/usr/bin/python

# cityscapes imports
from datasets.cityscapes.annotation import Annotation
from datasets.cityscapes.labels import name2label


# Convert the given annotation to a label image
def _get_instances(annotation, encoding):
    instances = []

    # loop over all objects
    for obj in annotation.objects:
        label = obj.label
        polygon = obj.polygon

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        if (label not in name2label) and label.endswith('group'):
            label = label[:-len('group')]

        if label not in name2label:
            print("Error: Label '{}' not known.".format(label))
            raise NotImplementedError

        # the label tuple
        label_tuple = name2label[label]

        if label_tuple.hasInstances:
            # get the class ID
            if encoding == "ids":
                id = label_tuple.id
            elif encoding == "trainIds":
                id = label_tuple.trainId
            else:
                print("Error: Encoding '{}' not known.".format(encoding))
                raise NotImplementedError

            instances.append((polygon, id))

    return instances


# Reads labels as polygons in JSON format and return the instance polygons with labels.
# json_fn is the filename of the json file
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
def json2instances(json_fn, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(json_fn)
    return _get_instances(annotation, encoding)
