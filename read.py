import lmdb
from caffe.io import caffe_pb2, datum_to_array
import cv2 as cv

env = lmdb.open("mnist_train_lmdb")
txn = env.begin()
cur = txn.cursor()
# print type(cur)
for key, value in cur:
    print(type(key), key)
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value)

    label = datum.label
    data = datum_to_array(datum)
    print data.shape
    print datum.channels
    image = data[0]
    # image = data.transpose(1, 2, 0)

    print(type(label))
    cv.imshow(str(label), image)
    cv.waitKey(0)

cv.destroyAllWindows()
env.close()