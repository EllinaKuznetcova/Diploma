__author__ = 'ellinakuznecova'
import cv2
import operator
from collections import defaultdict
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import image_names
import random
import scipy as sp
from datetime import datetime
from sys import argv

t1 = datetime.now()
det_vectors = []
key_points = []
frames_in_query = []
query_clusters = []
dir = os.path.dirname(__file__)

#parameters
MSCLUSTER_NUM = 10000
QUERY_IMGS_COUNT = 1
IMAGES_PATH = dir + '/kazan_attractions/'
threshold = 7
max_iter_num = 1000
min_clusters_for_frame = 7
val_point_x = 2
val_point_y = 2
num_of_img_apply_to = 10
result_images_count = 5
enable_query_expansion = False

query_image_path = 'kazan_attractions/chasha/9.jpg'

f1 = open('image_doc_kazan.txt', 'w+')

sift = cv2.SIFT(500)

possible_frames_kp = defaultdict(int)
normalization_count = [1] * MSCLUSTER_NUM

doc = [0] * MSCLUSTER_NUM
rejected_clusters = joblib.load('rejected_clusters_kazan.pkl')


def get_det_vectors(frame):
    global det_vectors
    kp_frame, det_vectors = sift.detectAndCompute(frame, None)
    for i in range(len(kp_frame)):
        key_points.append(kp_frame[i].pt)


def get_doc():
    global doc
    kmeans = joblib.load('kmeans_kazan.pkl')
    idf = joblib.load('idf_for_clusters_kazan.pkl')
    for i in range(MSCLUSTER_NUM):
        doc[i] = [0, []]
    print "started getting doc"
    for descr_num in range(len(det_vectors)):
        cluster_num = kmeans.predict(det_vectors[descr_num])[0]
        if cluster_num not in rejected_clusters:
            doc[cluster_num][0] += [item for item in idf if item[0] == cluster_num][0][1] * 1.0 / len(
                det_vectors)  # weighted count of word in doc
            doc[cluster_num][1].append(key_points[descr_num])

    for cluster_num in range(MSCLUSTER_NUM):
        if doc[cluster_num][0] != 0:
            print >> f1, cluster_num, ":", doc[cluster_num][1]
            query_clusters.append(cluster_num)
    print "finished getting doc"


def process_doc():
    print "started getting db doc"
    global possible_frames_kp
    possible_frames_kp = defaultdict(int)
    video_doc = open('document_kazan.txt', 'r')
    possible_frames = defaultdict(int)
    line = video_doc.readline()
    video_cluster_num = int(line[:line.find(':') - 1])
    for image_cluster_num in range(MSCLUSTER_NUM):
        if doc[image_cluster_num][0] != 0:
            while video_cluster_num < image_cluster_num:
                line = video_doc.readline()
                video_cluster_num = int(line[:line.find(':') - 1])
            if video_cluster_num == image_cluster_num:
                frame = int(line[line.find('[') + 2:line.find(',')])
                if frame not in frames_in_query:
                    frame_w = float(line[line.find(',') + 1:line.find(', [')])
                    line = line[line.find(', ['): len(line)]
                    pts = []
                    string_array = line[line.find(', ') + 1:line.find('],')]
                    while len(string_array) > 2:
                        pts.append((float(string_array[string_array.find('(') + 1: string_array.find(',')]),
                                    float(string_array[string_array.find(', ') + 1: string_array.find(')')])))
                        string_array = string_array[string_array.find(')') + 2:len(string_array)]
                    if frame not in possible_frames:
                        possible_frames[frame] = doc[image_cluster_num][0] * frame_w
                        possible_frames_kp[frame] = []
                        possible_frames_kp[frame].append([image_cluster_num, frame_w, pts])
                    else:
                        possible_frames[frame] += doc[image_cluster_num][0] * frame_w
                        possible_frames_kp[frame].append([image_cluster_num, frame_w, pts])
                line = line[line.find(']]') + 3:len(line)]
                while len(line) > 1:
                    frame = int(line[2:line.find(',')])
                    if frame not in frames_in_query:
                        frame_w = float(line[line.find(',') + 1:line.find(', [')])
                        line = line[line.find(', ['): len(line)]
                        pts = []
                        string_array = line[line.find(', ') + 1:line.find('],')]
                        while len(string_array) > 2:
                            pts.append((float(string_array[string_array.find('(') + 1: string_array.find(',')]),
                                        float(string_array[string_array.find(', ') + 1: string_array.find(')')])))
                            string_array = string_array[string_array.find(')') + 2:len(string_array)]
                        if frame not in possible_frames:
                            possible_frames[frame] = doc[image_cluster_num][0] * frame_w
                            possible_frames_kp[frame] = []
                            possible_frames_kp[frame].append([image_cluster_num, frame_w, pts])
                        else:
                            possible_frames[frame] += doc[image_cluster_num][0] * frame_w
                            possible_frames_kp[frame].append([image_cluster_num, frame_w, pts])
                    line = line[line.find(']]') + 3:len(line)]
    possible_frames = sorted(possible_frames.items(), key=operator.itemgetter(1))
    possible_frames.reverse()
    # print possible_frames
    possible_frames_kp = sorted(possible_frames_kp.items(), key=operator.itemgetter(0))
    # print possible_frames_kp
    print "finished getting db_doc"
    if enable_query_expansion :
        if len(frames_in_query) == 0:
            spacial_consistency(possible_frames)
        else:
            show_best_suitable_frame(possible_frames)
    else:
        show_best_suitable_frame(possible_frames)



def normal_spacial_consistency(frame):  # (n,[[cl,w,[pts]]])
    best_pts_img = ()
    best_pts_frame = ()
    validated_points = []

    query_clusters_for_frame = query_clusters[:]

    for cluster in query_clusters:
        if len([item for item in frame[1] if item[0] == cluster]) == 0:
            query_clusters_for_frame.remove(cluster)
    iteration = 0
    M_weight = 0
    best_M = 0
    if float(len(query_clusters_for_frame))/len(query_clusters) > 0.8:
        while M_weight <= threshold and iteration < max_iter_num and len(query_clusters_for_frame) > min_clusters_for_frame:
            cluster0 = query_clusters_for_frame[random.randint(0, len(query_clusters_for_frame) - 1)]
            query_clusters_for_frame.remove(cluster0)
            cluster1 = query_clusters_for_frame[random.randint(0, len(query_clusters_for_frame) - 1)]
            query_clusters_for_frame.remove(cluster1)
            cluster2 = query_clusters_for_frame[random.randint(0, len(query_clusters_for_frame) - 1)]
            query_clusters_for_frame.remove(cluster2)

            first_index1     = random.randint(0,len(doc[cluster0][1]) - 1)
            sec_index1       = random.randint(0,len(doc[cluster1][1]) - 1)
            third_index1     = random.randint(0,len(doc[cluster2][1]) - 1)

            pts1 = np.float32([doc[cluster0][1][first_index1], doc[cluster1][1][sec_index1], doc[cluster2][1][third_index1]])

            first_index2     = random.randint(0,len([item for item in frame[1] if item[0] == cluster0][0][2]) - 1)
            sec_index2       = random.randint(0,len([item for item in frame[1] if item[0] == cluster1][0][2]) - 1)
            third_index2     = random.randint(0,len([item for item in frame[1] if item[0] == cluster2][0][2]) - 1)
            pts2 = np.float32([[item for item in frame[1] if item[0] == cluster0][0][2][first_index2],
                               [item for item in frame[1] if item[0] == cluster1][0][2][sec_index2],
                               [item for item in frame[1] if item[0] == cluster2][0][2][third_index2]])

            M = cv2.getAffineTransform(pts1, pts2)
            best_M = M
            M_weight = 0
            validated_points = []  # [tr_point, pt, w, query_cl, db_cl]

            for cl in range(len(doc)):
                for point_num in range(len(doc[cl][1])):
                    tr_point = np.matrix(M) * np.float32([[doc[cl][1][point_num][0]], [doc[cl][1][point_num][1]], [1]])
                    tr_point = (float(tr_point[0]), float(tr_point[1]))
                    query_point = (float(doc[cl][1][point_num][0]), float(doc[cl][1][point_num][1]))

                    for doc_pts_for_cl in frame[1]:
                        for pt in doc_pts_for_cl[2]:
                            if abs(tr_point[0] - pt[0]) < val_point_x \
                                    and abs(tr_point[1] - pt[1]) < val_point_y:
                                M_weight += doc_pts_for_cl[1]
                                validated_points.append([query_point, pt, doc_pts_for_cl[1], cl, doc_pts_for_cl[0]])
                                if cl == doc_pts_for_cl[0]:
                                    M_weight += 1

            M_weight -= doc[cluster0][0] + doc[cluster1][0] + doc[cluster2][0]
            # print "weight: " + str(M_weight)
            iteration += 1

            if M_weight > threshold:
                best_pts_img = pts1
                best_pts_frame = pts2
            else:
                query_clusters_for_frame.append(cluster0)
                query_clusters_for_frame.append(cluster1)
                query_clusters_for_frame.append(cluster2)
    else:
        all_possible_combinations = []
        for cl0num in range(len(query_clusters_for_frame)):
            for cl1num in range(cl0num + 1,len(query_clusters_for_frame)):
                for cl2num in range(cl1num + 1,len(query_clusters_for_frame)):
                    cl0 = query_clusters_for_frame[cl0num]
                    cl1 = query_clusters_for_frame[cl1num]
                    cl2 = query_clusters_for_frame[cl2num]
                    all_possible_combinations.append([cl0,cl1,cl2])

        while M_weight <= threshold \
                and len(all_possible_combinations) > 0 \
                and len(query_clusters_for_frame) > min_clusters_for_frame \
                and iteration <= max_iter_num:
            clusters_index = random.randint(0,len(all_possible_combinations) - 1)
            clusters = all_possible_combinations[clusters_index]
            cluster0 = clusters[0]
            cluster1 = clusters[1]
            cluster2 = clusters[2]
            pts1 = np.float32([doc[cluster0][1][0], doc[cluster1][1][0], doc[cluster2][1][0]])
            pts2 = np.float32([[item for item in frame[1] if item[0] == cluster0][0][2][0],
                               [item for item in frame[1] if item[0] == cluster1][0][2][0],
                               [item for item in frame[1] if item[0] == cluster2][0][2][0]])

            M = cv2.getAffineTransform(pts1, pts2)
            best_M = M
            M_weight = 0
            validated_points = []  # [tr_point, pt, w, query_cl, db_cl]

            for cl in range(len(doc)):
                for point_num in range(len(doc[cl][1])):
                    tr_point = np.matrix(M) * np.float32([[doc[cl][1][point_num][0]], [doc[cl][1][point_num][1]], [1]])
                    tr_point = (float(tr_point[0]), float(tr_point[1]))
                    query_point = (float(doc[cl][1][point_num][0]), float(doc[cl][1][point_num][1]))

                    for doc_pts_for_cl in frame[1]:
                        for pt in doc_pts_for_cl[2]:
                            if abs(tr_point[0] - pt[0]) < val_point_x \
                                    and abs(tr_point[1] - pt[1]) < val_point_y:
                                M_weight += doc_pts_for_cl[1]
                                validated_points.append([query_point, pt, doc_pts_for_cl[1], cl, doc_pts_for_cl[0]])
                                if cl == doc_pts_for_cl[0]:
                                    M_weight += 1

            M_weight -= doc[cluster0][0] + doc[cluster1][0] + doc[cluster2][0]
            all_possible_combinations.remove(clusters)
            iteration += 1

            if M_weight > threshold:
                best_pts_img = pts1
                best_pts_frame = pts2

    if M_weight > threshold:
        return M_weight, best_M, best_pts_img, best_pts_frame, validated_points
    else:
        return 0, 0, 0, 0, 0


def spacial_consistency(possible_frames):
    possible_frames = possible_frames[0:num_of_img_apply_to]

    pos_frames_inliers = []
    print 'frames candidates to add in query:'
    for frame_data in possible_frames:

        frame_kp = [item for item in possible_frames_kp if item[0] == frame_data[0]][0]
        M_weight, M, pts_img, pts_doc, validated_points = normal_spacial_consistency(frame_kp)
        if validated_points != 0:
            pos_frames_inliers.append([frame_data[0], validated_points, len(validated_points)])
            print get_image_name(frame_data[0])
            # frame = cv2.imread(get_image_name(frame_data[0]),1)
            # plt.figure()
            # plt.imshow(frame)
            # plt.show()

    pos_frames_inliers = sorted(pos_frames_inliers, key=operator.itemgetter(2))  # sort frames by number of inliers
    pos_frames_inliers.reverse()
    pos_frames_inliers = pos_frames_inliers[0:QUERY_IMGS_COUNT]
    print 'frames added to query (' + str(QUERY_IMGS_COUNT) + '):'
    for frame in pos_frames_inliers:  # expand query
        frame_num = frame[0]
        print get_image_name(frame_num)
        frames_in_query.append(frame_num)
        # find points that matches and expand query with them
        matched_pts_for_frame = frame[1]
        for pts in matched_pts_for_frame:  # [tr_point, pt, w, query_cl, db_cl]
            cluster_num = pts[4]
            doc[cluster_num][0] += pts[2]
            normalization_count[cluster_num] += 1

    for cluster_num in range(MSCLUSTER_NUM):
        if normalization_count[cluster_num] != 1:
            doc[cluster_num][0] /= normalization_count[cluster_num]

    process_doc()


def match(frame, frame_num, index):
    frame_kp = [item for item in possible_frames_kp if item[0] == frame_num][0]
    M_weight, best_M, pts_img, pts_frame, validated_points = normal_spacial_consistency(frame_kp)
    if not isinstance(best_M,int):
        # h, w = img.shape
        # botton_left = np.matrix(best_M) * np.float32([[0], [0], [1]])
        # top_left = np.matrix(best_M) * np.float32([[0], [h - 1], [1]])
        # top_right = np.matrix(best_M) * np.float32([[w - 1], [h - 1], [1]])
        # bottom_right = np.matrix(best_M) * np.float32([[w - 1], [0], [1]])
        # color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        #
        # cv2.line(frame, (int(botton_left[0]), int(botton_left[1])),
        #          (int(top_left[0]), int(top_left[1])), color, 5)
        #
        # cv2.line(frame, (int(top_left[0]), int(top_left[1])),
        #          (int(top_right[0]), int(top_right[1])), color, 5)
        #
        # cv2.line(frame, (int(top_right[0]), int(top_right[1])),
        #          (int(bottom_right[0]), int(bottom_right[1])), color, 5)
        #
        # cv2.line(frame, (int(bottom_right[0]), int(bottom_right[1])),
        #          (int(botton_left[0]), int(botton_left[1])), color, 5)

        h1, w1 = img.shape[:2]
        h2, w2 = frame.shape[:2]
        view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
        view[:h1, :w1, 0] = img
        view[:h2, w1:] = frame
        view[:, :, 1] = view[:, :, 0]
        view[:, :, 2] = view[:, :, 0]

        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(view, (int(pts_img[0][0]), int(pts_img[0][1])),
                     (int(pts_frame[0][0] + w1), int(pts_frame[0][1])), color, 4)
        cv2.line(view, (int(pts_img[1][0]), int(pts_img[1][1])),
                     (int(pts_frame[1][0] + w1), int(pts_frame[1][1])), color, 4)
        cv2.line(view, (int(pts_img[2][0]), int(pts_img[2][1])),
                     (int(pts_frame[2][0] + w1), int(pts_frame[2][1])), color, 4)

        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        print "validated pts: " + str(len(validated_points))
        for matched_pts in validated_points:
            pt_img = matched_pts[0]
            pt_frame = matched_pts[1]
            cv2.line(view, (int(pt_img[0]), int(pt_img[1])),
                     (int(pt_frame[0] + w1), int(pt_frame[1])), color, 2)
        plt.figure()
        plt.imshow(view)

        cv2.imwrite("matched_kazan" + str(index + 1) + ".jpg", view)
    else:
        print "not enough matches found"

def get_image_num_string(number):
    result = str(number)
    while len(result) < 6:
        result = '0' + result
    result = '_' + result
    return result

def get_image_name(frame_num):
    return IMAGES_PATH + image_names.get_image_name(frame_num + 1)

def show_best_suitable_frame(possible_frames):
    for i in range(result_images_count):
        if i < len(possible_frames):
            suitable_frame_num = possible_frames[i][0]
            image_name = get_image_name(suitable_frame_num)
            print "result image " + str(i + 1) + " name: " + image_name
            frame = cv2.imread(image_name)
            match(frame, suitable_frame_num, i)
        else:
            print "number of found images less than " + str(i)
            break

if len(argv) > 1:
    script, result_images_count, enable_query_expansion, query_image_path = argv
result_images_count = int(result_images_count)
if not isinstance(enable_query_expansion,bool):
    enable_query_expansion = True if enable_query_expansion == "True" else False
    
img = cv2.imread(query_image_path, 0)
get_det_vectors(img)
get_doc()
process_doc()

plt.show()
t2 = datetime.now()
delta = t2 - t1
combined = delta.seconds + delta.microseconds/1E6
print "time: " + str(combined)