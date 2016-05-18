import cv2
import operator
from sklearn.cluster import KMeans
from math import log
from sklearn.externals import joblib
import os
import image_names
from datetime import datetime

t1 = datetime.now()
dir = os.path.dirname(__file__)
MSCLUSTER_NUM = 10000
IMAGES_PATH = dir + '/kazan_attractions/'
det_vectors = []
det_vectors_for_clustering = []
key_points = []
rejected_clusters = []
num_of_words_in_cluster = [0] * MSCLUSTER_NUM
for i in range(MSCLUSTER_NUM):
    num_of_words_in_cluster[i] = (i, 0)

f1 = open('document_kazan.txt', 'w+')

time_file = open('times','w+')

def get_det_vectors(frame, isForClustering = False):
    # detect keypoints
    kp1 = sift.detect(frame)

    # print '#keypoints in frame: %d' % (len(kp1))
    if len(kp1) > 0:
        # descriptors
        k1, d1 = sift.compute(frame, kp1)  # k1 - koord d1 - descr
        # print k1.pt
        det_vectors.append(d1)
        kp_array = []
        for i in range(len(k1)):
            kp_array.append(k1[i].pt)
        key_points.append(kp_array)
        if isForClustering:
            for i in range(len(d1)):
                det_vectors_for_clustering.append(d1[i])


def reject_freq_words():
    global num_of_words_in_cluster
    sorted_num = sorted(num_of_words_in_cluster, key=operator.itemgetter(1))
    num_of_top_5_pct_ms = int(MSCLUSTER_NUM * 0.05)
    num_of_bottom_10_pct_ms = int(MSCLUSTER_NUM * 0.1)
    for i in range(num_of_bottom_10_pct_ms):
        print sorted_num[0]
        rejected_clusters.append(sorted_num[0][0])
        sorted_num.remove(sorted_num[0])
    sorted_num.reverse()
    for i in range(num_of_top_5_pct_ms):
        print sorted_num[0]
        rejected_clusters.append(sorted_num[0][0])
        sorted_num.remove(sorted_num[0])
    joblib.dump(rejected_clusters,'rejected_clusters_kazan.pkl')
    num_of_words_in_cluster = sorted(sorted_num, key=operator.itemgetter(0))
    print 'number of clusters left: ', len(num_of_words_in_cluster)


def get_docs_from_file():
    global num_of_words_in_cluster

    # clustering
    print "started clustering"
    kmeans = KMeans(n_clusters=MSCLUSTER_NUM)
    kmeans.fit(det_vectors_for_clustering)
    joblib.dump(kmeans, 'kmeans_kazan.pkl')
    # kmeans = joblib.load('kmeans_kazan.pkl')
    print "finished clustering"

    doc1 = [0] * len(det_vectors)  # number of frames-docs
    print "started getting doc"
    for doc_number in range(len(det_vectors)):  # num of frame
        doc1[doc_number] = [0] * MSCLUSTER_NUM  # number of words in doc doc_number
        for i in range(MSCLUSTER_NUM):
            doc1[doc_number][i] = [0, []]
        for descr_num in range(len(det_vectors[doc_number])):
            cluster_num = kmeans.predict(det_vectors[doc_number][descr_num])
            doc1[doc_number][cluster_num][0] += 1  # count of word in doc(frame) doc_number
            doc1[doc_number][cluster_num][1].append(key_points[doc_number][descr_num])
            # equivalence of num_of_words_in_cluster[cluster_num] += 1 if num_of_words_in_cluster consist of tuples
            cluster_tuple = [item for item in num_of_words_in_cluster if item[0] == cluster_num][0]
            index = num_of_words_in_cluster.index(cluster_tuple)
            temp_list = list([item for item in num_of_words_in_cluster if item[0] == cluster_num][0])
            temp_list[1] += 1
            num_of_words_in_cluster[index] = tuple(temp_list)
    print "finished getting doc"

    sum = 0
    for q in range(len(doc1)):
        sum += doc1[q][0][0]
    print 'sum for 0 cluster: ', sum

    print num_of_words_in_cluster

    reject_freq_words()
    idf_for_clusters = map(lambda x: (x[0],log(len(det_vectors)*1.0 / x[1])), num_of_words_in_cluster)
    joblib.dump(idf_for_clusters, 'idf_for_clusters_kazan.pkl')
    for i in range(len(det_vectors)):  # num of frame
        num_of_w_in_cluster_iterator = 0
        while num_of_w_in_cluster_iterator < len(num_of_words_in_cluster):
            cluster_num = num_of_words_in_cluster[num_of_w_in_cluster_iterator][0]
            if doc1[i][cluster_num][0] != 0:
                doc1[i][cluster_num][0] = doc1[i][cluster_num][0] * 1.0 / len(det_vectors[i]) * \
                                          [item for item in idf_for_clusters if item[0] == cluster_num][0][1]
            else:
                doc1[i][cluster_num][0] = 0.0
            num_of_w_in_cluster_iterator += 1

    for cluster_num in range(MSCLUSTER_NUM):
        if cluster_num not in rejected_clusters:
            cluster_list = []
            for frame_num in range(len(doc1)):
                if doc1[frame_num][cluster_num][0] != 0.0:
                    temp_list = [frame_num, doc1[frame_num][cluster_num][0], doc1[frame_num][cluster_num][1]]
                    cluster_list.append(temp_list)
            print >> f1, cluster_num, ":", cluster_list


def images_processor():
    for building_num in range(len(image_names.titles)):
        image_num = 0
        while image_num <= image_names.max_photos_num[building_num]:
            image = cv2.imread(IMAGES_PATH + image_names.titles[building_num] + '/'+ str(image_num + 1) + '.jpg', 0)
            if image is not None:
                print image_names.titles[building_num] + ' ' + str(image_num)
                if image_num <= image_names.max_photos_num[building_num]*(0.67):
                    get_det_vectors(image,True)
                else:
                    get_det_vectors(image)
            image_num += 1
    print "finished processing images"



sift = cv2.SIFT(500)
images_processor()
get_docs_from_file()
cv2.destroyAllWindows()
t2 = datetime.now()
delta = t2 - t1
combined = delta.seconds + delta.microseconds/1E6
print "time: " + str(combined)
print >> time_file, str(combined)
