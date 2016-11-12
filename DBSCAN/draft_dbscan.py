import skimage.io
import skimage.draw
import skimage.transform
import math
import queue

class Point:

    def __init__(self, new_x, new_y):

        self.x = new_x
        self.y = new_y

class Detection:

    def __init__(self, new_top_left_corner, new_width, new_height, new_score=1):

        self.top_left_corner = new_top_left_corner
        self.width = new_width
        self.height = new_height
        self.score = new_score
        self.clustering_status = 0 # 0 : not visited yet, -1 : noize, else: cluster index

def load_detections():

    detections = []
    f = open('detections_for_clustering_test.txt', 'r')
    for line in f:
        if line.strip():
            words = line.split(' ')
            detections.append(Detection(Point(int(float(words[0])), int(float(words[1]))), int(float(words[2])), int(float(words[3])), int(float(words[4]))))
    return detections

def draw_detections(detections, infilename, outfilename):

    image = skimage.io.imread(infilename, as_grey=True)

    for detection in detections:
        #we get the coordinates of the square, and we display a white square on the image (never managed to make draw polygon work)
        rr, cc = skimage.draw.line(detection.top_left_corner.x, detection.top_left_corner.y, detection.top_left_corner.x+detection.width, detection.top_left_corner.y)
        image[rr,cc] = 1
        rr, cc = skimage.draw.line(detection.top_left_corner.x+detection.width, detection.top_left_corner.y, detection.top_left_corner.x+detection.width, detection.top_left_corner.y+detection.height)
        image[rr,cc] = 1
        rr, cc = skimage.draw.line(detection.top_left_corner.x, detection.top_left_corner.y+detection.height, detection.top_left_corner.x+detection.width, detection.top_left_corner.y+detection.height)
        image[rr,cc] = 1
        rr, cc = skimage.draw.line(detection.top_left_corner.x, detection.top_left_corner.y, detection.top_left_corner.x, detection.top_left_corner.y+detection.height)
        image[rr,cc] = 1
    
    skimage.io.imsave(outfilename, image)

def draw_grouped_detections(clusters, in_source_filename, in_detections_filename, out_clusters_on_source_filename, out_clusters_on_detections_filname):

    image = skimage.io.imread(in_source_filename, as_grey=True)
    image2 = skimage.io.imread(in_detections_filename, as_grey=True)

    for cluster_num, detection in clusters.items():
        x = int(detection[0]+0.5*detection[2])
        y = int(detection[1]+0.5*detection[3])
        half_diag = int(0.5*math.sqrt(math.pow(detection[2], 2)+math.pow(detection[3],2)))
        rr, cc = skimage.draw.circle_perimeter(x, y, half_diag)
        image[rr, cc] = 1
        image2[rr, cc] = 255

    skimage.io.imsave(out_clusters_on_source_filename, image)
    skimage.io.imsave(out_clusters_on_detections_filname, image2)

def group_detections(detections, min_score):

    # append detections into clusters map { cluster_number => [detection1, ... detectionN] }
    clusters = {}
    nb_clusters = 0
    for detection in detections:
        if detection.clustering_status > 0 and detection.clustering_status in clusters:
            clusters[detection.clustering_status].append(detection)
        elif detection.clustering_status > 0:
            nb_clusters += 1
            clusters[detection.clustering_status] = []
            clusters[detection.clustering_status].append(detection)

    # for each cluster, group detections into an average    
    grouped_detections = {} 
    for i in range(0, nb_clusters):
        detections = clusters[i+1]
        avg_x = 0
        avg_y = 0
        avg_width = 0
        avg_height = 0
        score_sum = 0
        for detection in detections:
            score_delta = math.pow((detection.score-min_score)*100, 2) # substract min_score and pow(,2) to emphasize differences (make high score matter even more)
            # make all relevant values average
            avg_x += score_delta*detection.top_left_corner.x
            avg_y += score_delta*detection.top_left_corner.y
            avg_width += score_delta*detection.width
            avg_height += score_delta*detection.height
            score_sum += score_delta
        avg_x /= score_sum
        avg_y /= score_sum
        avg_width /= score_sum
        avg_height /= score_sum
        grouped_detections[i+1] = (int(avg_x), int(avg_y), int(avg_width), int(avg_height))
    return grouped_detections

def DBSCAN(detections, epsilon, min_pts):

    c = 0 # cluster index, begining at 1
    for detection in detections:
        if detection.clustering_status != 0: # already visited
            pass
        else: # not visited yet
            neighbors = find_neighbors(detections, detection, epsilon)

            if len(neighbors) < min_pts: # has not enough neighbors to compose a cluster
                detection.clustering_status = -1
            else:
                c += 1 # Start a new cluster
                extend_cluster(detections, detection, neighbors, c, epsilon, min_pts)

def extend_cluster(detections, detection, neighbors, c, epsilon, min_pts):

    # mark as belonging to cluster C
    detection.clustering_status = c

    not_visited_neighbors = queue.Queue() # queue of not visited neighbors
    for neighbor in neighbors:
        if neighbor.clustering_status == 0: # not visited yet
            not_visited_neighbors.put(neighbor)

    while not not_visited_neighbors.empty(): # while there is reachable neighbors
        neighbor = not_visited_neighbors.get()
        neighbor.clustering_status = c # add to current cluster
        extra_neighbors = list(set(find_neighbors(detections, neighbor, epsilon)) - set(neighbors))
        if len(extra_neighbors) >= min_pts:
            for extra_neighbor in extra_neighbors:
                if extra_neighbor.clustering_status == 0: #not visited yet
                    not_visited_neighbors.put(extra_neighbor) # find more reachable neighbors

def find_neighbors(detections, center_detection, epsilon):

    epsilon_neighbors = []
    for detection in detections:
        dist = distance(detection, center_detection)
        if dist <= epsilon:
            epsilon_neighbors.append(detection)
    return epsilon_neighbors

def distance(detection1, detection2):

    # Rules : distance < 36 : cannot be two detectable separated faces because detector can only detect 36*36 or bigger

    d1_centroid_x = detection1.top_left_corner.x+0.5*detection1.width
    d1_centroid_y = detection1.top_left_corner.y+0.5*detection1.height
    d2_centroid_x = detection2.top_left_corner.x+0.5*detection2.width
    d2_centroid_y = detection2.top_left_corner.y+0.5*detection2.height

    # Euclidian distance
    distance = math.sqrt(math.pow(d2_centroid_x-d1_centroid_x,2)+math.pow(d2_centroid_y-d1_centroid_y,2)+math.pow((detection2.width-detection1.width),2)+math.pow((detection2.height-detection1.height),2))
    '''
    # Calculate overlap
    max_left = max(detection1.top_left_corner.x, detection2.top_left_corner.x)
    min_right = min(detection1.top_left_corner.x+detection1.width, detection2.top_left_corner.x+detection2.width)
    max_top = max(detection1.top_left_corner.y, detection2.top_left_corner.y)
    min_bottom = min(detection1.top_left_corner.y+detection1.height, detection2.top_left_corner.y+detection2.height)

    if(max_left < min_right and max_top < min_bottom):
        #overlapping
        
        #calculate overalapping ratio
        overlap_area = (min_bottom-max_top)*(min_right-max_left)
        d1_area = detection1.width*detection1.height
        d2_area = detection2.width*detection2.height
        overlapping_ratio = overlap_area/(d1_area+d2_area-overlap_area)
        distance = 1-overlapping_ratio
        print(overlapping_ratio)
    else:
        distance = -1 #represents infinite
    #print(distance)
    '''
    return distance

##################################### Tests

detections = load_detections()
DBSCAN(detections, 36, 2)
grouped_detections = group_detections(detections, 0.99)
draw_detections(detections, 'merica.jpg', 'merica_with_detections.jpg')
draw_grouped_detections(grouped_detections, 'merica.jpg', 'merica_with_detections.jpg', 'merica_with_clusters_on_source.jpg', 'merica_with_clusters_on_detections.jpg')
