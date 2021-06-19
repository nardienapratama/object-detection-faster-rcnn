# REFERENCE: ImageNetBundle Textbook (PyImageSearch)
# import the necessary packages
from config import synthetic_ds_config as config
from tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2
import json
import fnmatch

# REFERENCE: https://stackoverflow.com/questions/52879192/merge-multiple-json-files-more-than-two/52879638
def merge_JSONFiles(filenames):
    print("MERGING JSON")
    output_list = list()
    new_merged_json = "synthetic_ds/merged_captures.json"
    for f1 in filenames:
        with open(f1, 'rb') as infile:
            output_list.append(json.load(infile))

    all_items = []
    for json_file in output_list:
        all_items += json_file['captures']

    with open('synthetic_ds/merged_captures.json', 'w') as textfile_merged:
        json.dump({"captures": all_items }, textfile_merged)

def main(_):
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")

    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                    "\tid: " + str(v) + "\n"
                    "\tname: '" + k + "'\n"
                    "}\n")
        f.write(item)

    # close the output classes file
    f.close()

    # initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    # rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header
    # for row in rows[1:]:
    #     # break the row into components
    #     row = row.split(",")[0].split(";")
    #     (imagePath, label, startX, startY, endX, endY, _) = row
    #     (startX, startY) = (float(startX), float(startY))
    #     (endX, endY) = (float(endX), float(endY))

    #     # if we are not interested in the label, ignore it
    #     if label not in config.CLASSES:
    #         continue

    #     # build the path to the input image, then grab any other
    #     # bounding boxes + labels associated with the image
    #     # path, labels, and bounding box lists, respectively
    #     p = os.path.sep.join([config.BASE_PATH, imagePath])
    #     b = D.get(p, [])

    #     # build a tuple consisting of the label and bounding box,
    #     # then update the list and store it in the dictionary
    #     b.append((label, (startX, startY, endX, endY)))
    #     D[p] = b

    ITERATION = 1

    # TODO: MERGE ALL JSON FILES FIRST

    merged_json_file_path = "synthetic_ds/merged_captures.json"

    if os.path.exists(merged_json_file_path):
        os.remove(merged_json_file_path)

    # put all captures.json files in a list
    jsonFilesList = ["synthetic_ds/" + f for f in os.listdir('synthetic_ds') if f.endswith('.json')]
    # print(["synthetic_ds/" + filename for filename in os.listdir('synthetic_ds') if fnmatch.fnmatch(filename, '[captures]*.json')])
    print(jsonFilesList)

    # create new merged_captures.json which has contents of all capture json files
    merge_JSONFiles(jsonFilesList)

    

    with open(merged_json_file_path, 'r') as j:
        contents = json.loads(j.read())

    for image in contents['captures']:
        print(len(contents['captures']))
        imagePath = "dataset/" + image['filename'].split('/')[-1] # just the rgb part of the filename
        # print("CHECK imagepath:", imagePath)
        for annotation in image['annotations'][0]['values']:
            label = annotation['label_name']
            startX = annotation['x']
            startY = annotation['y']
            endX = annotation['x'] + annotation['width']
            endY = annotation['y']+ annotation['height']
            (startX, startY) = (float(startX), float(startY))
            (endX, endY) = (float(endX), float(endY))

            # if we are not interested in the label, ignore it
            if label not in config.CLASSES:
                continue

            # build the path to the input image, then grab any other
            # bounding boxes + labels associated with the image
            # path, labels, and bounding box lists, respectively
            p = os.path.sep.join([config.BASE_PATH, imagePath])
            b = D.get(p, [])

            # build a tuple consisting of the label and bounding box,
            # then update the list and store it in the dictionary
            b.append((label, (startX, startY, endX, endY)))
            D[p] = b


    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
        test_size=config.TEST_SIZE, random_state=42)
    
    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
    ]

    # loop over the datasets (training and testing splits)
    for (dType, keys, outputPath) in datasets:
        # initialize the TensorFlow writer and initialize the total
        # number of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.io.TFRecordWriter(outputPath)
        total = 0

        # loop over all the keys in the current set
        for k in keys:
            # print("k:", k)
            # load the input image from disk as a TensorFlow object
            encoded = tf.io.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            # load the image from disk again, this time as a PIL
            # object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            # print("Filename:", filename)
            encoding = filename[filename.rfind(".") + 1:]

            # initialize the annotation object used to store
            # information regarding the bounding box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # loop over the bounding boxes + labels associated with
            # the image
            
            for (label, (startX, startY, endX, endY)) in D[k]:
                # print("For k: ", k, " --", (label, (startX, startY, endX, endY)))
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                if ITERATION <=10:
                    # load the input image from disk and denormalize the bounding box coordinates
                    image = cv2.imread(k)
                    startX = int(xMin * w)
                    startY = int(yMin * h)
                    endX = int(xMax * w)
                    endY = int(yMax * h)

                    # draw the bounding box on the image
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    # show the output image
                    # print("Image:", filename)
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
                    
                # print("Iteration ", ITERATION)

                # update the bounding boxes + labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                # increment the total number of examples
                total += 1
            ITERATION += 1
            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)

            # add the example to the writer
            writer.write(example.SerializeToString())

            # close the writer and print diagnostic information to the
            # user
        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total,
            dType))

# check to see if the main thread should be started
if __name__ == "__main__":
    tf.compat.v1.app.run()