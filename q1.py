import statistics
import time

import cv2
import imageio as imageio
import numpy as np


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def croping(size, pan):
    print('cropping started')
    panorama = cv2.VideoCapture(pan)
    width = int(panorama.get(3))
    height = int(panorama.get(4))
    w = int(width / size) + 1
    h = int(height / size) + 1
    s1 = w * size - width
    s2 = h * size - height
    # print(width, height, s1, s2, w, h)

    for j in range(h):
        if j == h - 1:
            size2 = s2
        else:
            size2 = 0
        panorama = cv2.VideoCapture(pan)
        res = [cv2.VideoWriter('vids/crop' + str(j) + '-' + str(i) + '.mp4', -1,
                               30, (size, size - size2)) for i in
               range(w - 1)]
        res.append(cv2.VideoWriter('vids/crop' + str(j) + '-' + str(w - 1) + '.mp4',
                                   -1, 30, (size - s1, size - size2)))
        res = np.array(res)

        for i in range(900):

            ret, frame = panorama.read()
            for x in range(w):
                if x == w - 1:
                    size1 = s1
                else:
                    size1 = 0
                # print(size1, size2)
                # print(j * size, '-', (j + 1) * size - size2, '/', x * size, '-', (x + 1) * size - size1)
                res[x].write(frame[j * size:(j + 1) * size - size2, x * size:(x + 1) * size - size1])
            # print(i, '-', j, '/', h - 1)

        cv2.destroyAllWindows()
        for x in range(w):
            res[x].release()
        print('-----Row', j, 'completed-----')


def find_homography(image1, image2):
    image1 = cv2.cvtColor(resize(image1, 1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(resize(image2, 1), cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, des1 = sift.detectAndCompute(image1, None)
    keypoints2, des2 = sift.detectAndCompute(image2, None)
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(des1, des2)
    print(len(matches))
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.int32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.int32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    print("RANSAC started")
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=15, maxIters=1000000000,
                                 confidence=0.99)

    return H


def fast_homography(image1, image2, itration):
    # print(p)
    sift = cv2.SIFT_create()
    keypoints1, des1 = sift.detectAndCompute(image1, None)
    keypoints2, des2 = sift.detectAndCompute(image2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.85 + (itration / 9000)
    matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            matches.append(m)
    print(len(matches))

    # matches = sorted(matches, key=lambda x: x.distance)[:1000]
    src_pts = np.int32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.int32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # print("RANSAC started")
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=25, maxIters=1000000000,
                                 confidence=0.99)
    # print("Homographic Matrix:")
    # print(H)

    return H


def part1(image1, image2):
    print('part1 started')
    H = find_homography(image1, image2)
    Mat = np.linalg.inv(H)
    h, w = image2.shape[:2]
    pts = np.float32([[[w / 8, 2 * h / 8]], [[w / 8, 5 * h / 8]], [[4 * w / 8, 5 * h / 8]], [[4 * w / 8, 2 * h / 8]]])
    src = np.array(pts, dtype='int')
    dst = np.array(cv2.perspectiveTransform(pts, Mat), dtype='int')
    img2 = image2.copy()
    img1 = image1.copy()

    for i in range(4):
        cv2.line(img2, src[i - 1][0], src[i][0], (0, 0, 255), 4)

    for i in range(4):
        cv2.line(img1, dst[i - 1][0], dst[i][0], (0, 0, 255), 4)

    cv2.imwrite('res01-450-rect.jpg', img2)
    cv2.imwrite('res02-270-rect.jpg', img1)

    corners = []
    c = [np.matmul(H, [0, 0, 1]), np.matmul(H, [0, h, 1]),
         np.matmul(H, [w, 0, 1]), np.matmul(H, [w, h, 1])]

    for i in range(4):
        corners.append([c[i][0] / c[i][2], c[i][1] / c[i][2]])

    corners = np.array(corners)

    left = int(np.min(corners[:, 1]))
    up = int(np.min(corners[:, 0]))
    right = int(np.max(corners[:, 1]))
    down = int(np.max(corners[:, 0]))

    height = down - up
    width = right - left

    shift = np.float32([
        [1, 0, -up],
        [0, 1, -left],
        [0, 0, 1]])

    H = np.matmul(shift, H)

    corners = []
    c = [np.matmul(shift, [0, 0, 1]), np.matmul(shift, [0, h, 1]),
         np.matmul(shift, [w, 0, 1]), np.matmul(shift, [w, h, 1])]

    for i in range(4):
        corners.append([c[i][0] / c[i][2], c[i][1] / c[i][2]])

    print(corners)
    corners = np.array(corners)

    down = int(np.max(corners[:, 1]))
    right = int(np.max(corners[:, 0]))

    print(down, height, right, width)
    down = max(down, width)
    right = max(right, height)

    plate1 = cv2.warpPerspective(image1, H, (right, down))
    plate2 = cv2.warpPerspective(image2, shift, (right, down))

    panorama = plate2.copy()

    print('creating panorama')
    for i in range(right):
        for j in range(down):
            if np.max(panorama[j, i]) == 0 and np.max(plate1[j, i]) != 0:
                panorama[j, i] = plate1[j, i]

    cv2.imwrite('res03-270-450-panorama.jpg', panorama)


def part2(im1, im2, im3, im4, im5):
    print('part2 started')
    h, w = im1.shape[:2]
    H = [0, 0, 0, 0]

    H[1] = find_homography(im2, im3)
    H[0] = np.matmul(find_homography(im1, im2), H[1])
    H[2] = find_homography(im4, im3)
    H[3] = np.matmul(find_homography(im5, im4), H[2])

    coords = [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]

    cal_coords = []

    for j in range(4):
        for i in range(4):
            res = np.matmul(H[i], coords[j])
            cal_coords.append([res[0] / res[2], res[1] / res[2]])

    cal_coords = np.array(cal_coords)

    up_limit = int(np.min(cal_coords[:, 1]))
    left_limit = int(np.min(cal_coords[:, 0]))
    down_limit = int(np.max(cal_coords[:, 1]))
    right_limit = int(np.max(cal_coords[:, 0]))

    print(up_limit, left_limit, down_limit, right_limit)

    shift = np.float32([
        [1, 0, -left_limit],
        [0, 1, -up_limit],
        [0, 0, 1]])
    H[0] = np.matmul(shift, H[0])
    H[1] = np.matmul(shift, H[1])
    H[2] = np.matmul(shift, H[2])
    H[3] = np.matmul(shift, H[3])

    width = right_limit - left_limit
    height = down_limit - up_limit

    plate1 = cv2.warpPerspective(im1, H[0], (width, height))
    plate2 = cv2.warpPerspective(im2, H[1], (width, height))
    plate3 = cv2.warpPerspective(im3, shift, (width, height))
    plate4 = cv2.warpPerspective(im4, H[2], (width, height))
    plate5 = cv2.warpPerspective(im5, H[3], (width, height))

    m1 = np.zeros((height, width), dtype='uint8')
    m2 = np.zeros((height, width), dtype='uint8')
    m3 = np.zeros((height, width), dtype='uint8')
    m4 = np.zeros((height, width), dtype='uint8')
    m5 = np.zeros((height, width), dtype='uint8')

    m1[cv2.cvtColor(cv2.medianBlur(plate1, 9), cv2.COLOR_BGR2GRAY) != 0] = 1
    m2[cv2.cvtColor(cv2.medianBlur(plate2, 9), cv2.COLOR_BGR2GRAY) != 0] = 1
    m3[cv2.cvtColor(cv2.medianBlur(plate3, 9), cv2.COLOR_BGR2GRAY) != 0] = 1
    m4[cv2.cvtColor(cv2.medianBlur(plate4, 9), cv2.COLOR_BGR2GRAY) != 0] = 1
    m5[cv2.cvtColor(cv2.medianBlur(plate5, 9), cv2.COLOR_BGR2GRAY) != 0] = 1

    plate = plate1.copy()
    kernel = np.ones((5, 5), np.uint8)
    m1 = cv2.erode(m1, kernel, iterations=1)
    m2 = cv2.erode(m2, kernel, iterations=1)
    m4 = cv2.erode(m4, kernel, iterations=1)
    m5 = cv2.erode(m5, kernel, iterations=1)

    double_mask = [np.array(m1 + m2, dtype='int'), np.array(m3 + m2, dtype='int'),
                   np.array(m3 + m4, dtype='int'), np.array(m5 + m4, dtype='int')]

    m1[m2 == 1] = 0
    m5[m4 == 1] = 0
    m2[m3 == 1] = 0
    m4[m3 == 1] = 0

    negative_point = -5

    for i in range(4):

        print('Draw masks and find points ... -', i)
        double_mask[i][double_mask[i] == 1] = negative_point
        ax0_up = np.argmax(np.sum(double_mask[i][:int(height / 2), :], axis=0))
        ax0_down = np.argmax(np.sum(double_mask[i][int(height / 2):, :], axis=0))

        for q in range(height):
            if double_mask[i][q, ax0_up] != 0:
                ax1_up = q
                break
        for q in range(height):
            if double_mask[i][height - q - 1, ax0_down] != 0:
                ax1_down = height - q - 1
                break

        for q in range(width):
            if np.max(double_mask[i][:, q]) == 2:
                left = q
                break
        for q in range(width):
            if np.max(double_mask[i][:, width - q - 1]) == 2:
                right = width - q - 1
                break

        double_mask[i][double_mask[i] == negative_point] = 1
        double_mask[i] = double_mask[i] * 127
        cv2.circle(double_mask[i], (ax0_up, ax1_up), 5, (200, 200, 200), 3)
        cv2.circle(double_mask[i], (ax0_down, ax1_down), 5, (200, 200, 200), 3)

        # cv2.imwrite('t' + str(i + 1) + '.jpg', double_mask[i])

        if i == 0:
            p1 = plate1
            p2 = plate2
        elif i == 1:
            p1 = plate2
            p2 = plate3
        elif i == 2:
            p1 = plate3
            p2 = plate4
        elif i == 3:
            p1 = plate4
            p2 = plate5

        print('Find best route ... -', i)
        check1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        check2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
        max = 255 ** 2 * height
        image_boundry = (check1 - check2) ** 2
        image_costs = np.zeros_like(image_boundry) + max
        image_costs[ax1_down, ax0_down] = 1
        visited = np.zeros_like(image_boundry)
        for row in range(2 + ax1_down, ax1_up - 1, -1):
            for clmn in range(right, left - 1, -1):
                if visited[row, clmn] == 0:
                    if clmn < left + 1:
                        cost = np.min(image_costs[row + 1, clmn:clmn + 2]) + image_boundry[row, clmn]
                    elif clmn > right - 1:
                        cost = np.min(image_costs[row + 1, clmn - 1:clmn + 1]) + image_boundry[row, clmn]

                    else:
                        cost = np.min(np.append(image_costs[row + 1, clmn - 1:clmn + 2],
                                                [image_costs[row, clmn - 1], image_costs[row, clmn + 1]])
                                      ) + image_boundry[row, clmn]
                    image_costs[row, clmn] = np.min([cost, image_costs[row, clmn]])
                    visited[row, clmn] = 1

        list = []
        list.append([ax1_up, ax0_up])
        row = ax1_up
        while True:
            row += 1
            if row == ax1_down:
                break
            min = np.infty
            done = False
            for clmn in range(-1, 4):

                if clmn < 2:
                    if clmn + list[- 1][1] < left or clmn + list[-1][1] > right:
                        cost = max
                    else:
                        cost = image_costs[row, clmn + list[-1][1]]
                elif clmn == 2:
                    if list[-1][1] + 1 < left or list[-1][1] + 1 > right:
                        cost = max
                    else:
                        cost = image_costs[row - 1, list[-1][1] + 1]
                else:
                    if list[-1][1] - 1 < left or list[-1][1] - 1 > right:
                        cost = max
                    else:
                        cost = image_costs[row - 1, list[-1][1] - 1]

                if cost < min:
                    min = cost
                    if clmn < 2:
                        minx = clmn + list[-1][1]
                    elif clmn == 2:
                        minx_inrow = list[-1][1] + 1
                        done = True
                    elif clmn == 3:
                        minx_inrow = list[-1][1] - 1
                        done = True
            if done:
                list.append([row - 1, minx_inrow])
                row -= 1
            else:
                list.append([row, minx])

            grad = 15
        print('Complete panorama ... -', i)
        ptest = plate1.copy()

        for r in range(len(list)):
            if i == 0:
                grad = int((r / len(list)) ** 1.6 * 120)
            cv2.circle(ptest, (list[r][1], list[r][0]), 1, (0, 0, 255), 2)
            l = list[r][1]
            r_prime = list[r][0]
            plate[r_prime, l + grad:] = p2[r_prime, l + grad:]

            for matte in range(l - grad, l):
                plate[r_prime, matte] = np.round(
                    p1[r_prime, matte] * ((grad + l - matte) / (grad * 2)) + p2[r_prime, matte] * (
                            (matte + grad - l) / (grad * 2)))

            for matte in range(l, l + grad):
                plate[r_prime, matte] = np.round(
                    p1[r_prime, matte] * ((grad + l - matte) / (grad * 2)) + p2[r_prime, matte] * (
                            (matte + grad - l) / (grad * 2)))
            if i > 1:
                for r in range(ax1_up):
                    plate[r, ax0_up:] = p2[r, ax0_up:]
                for r in range(ax1_down, height):
                    plate[r, ax0_down:] = p2[r, ax0_down:]

    cv2.imwrite('res04-key-frames-panorama.jpg', plate)


def part3(vid, key):
    print('part3 started')
    frames = 900
    H4 = find_homography(key[4], key[5])
    H3 = np.matmul(find_homography(key[3], key[4]), H4)
    H2 = np.matmul(find_homography(key[2], key[3]), H3)
    H1 = np.matmul(find_homography(key[1], key[2]), H2)
    H0 = np.matmul(find_homography(key[0], key[1]), H1)
    H5 = find_homography(key[6], key[5])
    H6 = np.matmul(find_homography(key[7], key[6]), H5)
    H7 = np.matmul(find_homography(key[8], key[7]), H6)
    H8 = np.matmul(find_homography(key[9], key[8]), H7)
    H9 = np.matmul(find_homography(key[10], key[9]), H8)

    print('part3 started')
    video = cv2.VideoCapture(vid)
    width = int(video.get(3))
    height = int(video.get(4))

    H = []
    for i in range(frames):
        ret, frame = video.read()

        if i <= 120:
            H.append(np.matmul(fast_homography(frame, key[1], i), H1))
        elif 120 < i <= 210:
            H.append(np.matmul(fast_homography(frame, key[2], i), H2))
        elif 210 < i <= 300:
            H.append(np.matmul(fast_homography(frame, key[3], i), H3))
        elif 300 < i <= 390:
            H.append(np.matmul(fast_homography(frame, key[4], i), H4))
        elif 390 < i <= 510:
            H.append(fast_homography(frame, key[5], i))
        elif 510 < i <= 600:
            H.append(np.matmul(fast_homography(frame, key[6], i), H5))
        elif 600 < i <= 690:
            H.append(np.matmul(fast_homography(frame, key[7], i), H6))
        elif 690 < i <= 780:
            H.append(np.matmul(fast_homography(frame, key[8], i), H7))
        elif 780 < i:
            H.append(np.matmul(fast_homography(frame, key[9], i), H8))

        print(i, '/ 899')
    np.save('Homography_unshifted.npy', H)
    coords = [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]

    cal_coords = []

    for j in range(4):
        for i in range(frames):
            res = np.matmul(H[i], coords[j])
            cal_coords.append([res[0] / res[2], res[1] / res[2]])

    cal_coords = np.array(cal_coords)

    up_limit = int(np.min(cal_coords[:, 1]))
    left_limit = int(np.min(cal_coords[:, 0]))
    down_limit = int(np.max(cal_coords[:, 1]))
    right_limit = int(np.max(cal_coords[:, 0]))

    print(up_limit, left_limit, down_limit, right_limit)

    shift = np.float32([
        [1, 0, -left_limit],
        [0, 1, -up_limit],
        [0, 0, 1]])

    for i in range(frames):
        H[i] = np.matmul(shift, H[i])

    width = right_limit - left_limit
    height = down_limit - up_limit

    np.save('Homography.npy', H)

    # result = cv2.VideoWriter('res05-reference-plane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,(width, height))
    result = imageio.get_writer('res05-reference-plane.mp4', fps=30)
    video = cv2.VideoCapture(vid)
    for i in range(frames):
        # p = int(i / 100)
        print(i)
        ret, frame = video.read()
        result.append_data(cv2.cvtColor(cv2.warpPerspective(frame, H[i], (width, height)), cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    result.close()


def part4(size, w_its, h_its, h, w):
    panorama = np.zeros((h, w, 3), dtype='uint8')
    print('part4 started')
    limit = 10
    # nearby = 30
    # i_piece = 4
    # j_piece = 9
    for i_piece in range(w_its):
        for j_piece in range(h_its):
            print(i_piece, '/', w_its - 1, ' - ', j_piece, '/', h_its - 1)
            # print('vids/crop' + str(i_piece) + '-' + str(j_piece) + '.mp4')
            croped = cv2.VideoCapture('vids/crop' + str(i_piece) + '-' + str(j_piece) + '.mp4')
            width = int(croped.get(3))
            height = int(croped.get(4))
            tmp = np.zeros((900, height, width, 3), dtype='uint8')
            res = np.zeros((height, width, 3), dtype='int')
            zero = np.zeros((height, width))
            blues = np.array([zero for i in range(900)], dtype='uint8')
            greens = np.array([zero for i in range(900)], dtype='uint8')
            reds = np.array([zero for i in range(900)], dtype='uint8')

            for i in range(900):
                tmp[i] = croped.read()[1]

            if np.sum(tmp) < 10000:
                print('skipped')
                continue

            for i in range(900):
                blues[i], greens[i], reds[i] = cv2.split(tmp[i])

            blues = np.array(blues, dtype='int')
            greens = np.array(greens, dtype='int')
            reds = np.array(reds, dtype='int')
            tmp = np.array(tmp, dtype='int')

            for j in range(width):
                # print(np.sum(tmp[:, :, j]))
                if np.sum(tmp[:, :, j]) < 1000:
                    # print(j)
                    continue
                for i in range(height):
                    # print(np.sum(tmp[:, i, j]))
                    if np.sum(tmp[:, i, j]) < 100:
                        continue
                    pixels_b = blues[:, i, j].copy()
                    pixels_g = greens[:, i, j].copy()
                    pixels_r = reds[:, i, j].copy()

                    boolean = (pixels_b > limit) + (pixels_g > limit) + (pixels_r > limit)
                    pixels_b = pixels_b[boolean]
                    pixels_g = pixels_g[boolean]
                    pixels_r = pixels_r[boolean]

                    if len(pixels_b) == 0 or len(pixels_g) == 0 or len(pixels_r) == 0:
                        # print()
                        continue

                    med_b = statistics.median(pixels_b)
                    med_g = statistics.median(pixels_g)
                    med_r = statistics.median(pixels_r)

                    res[i, j] = [med_b, med_g, med_r]
                    '''
                    for b in range(len(boolean)):
                        if not boolean[b]:
                            low = max(0, b - nearby)
                            high = min(len(boolean) - 1, b + nearby)
                            list_b[b] = statistics.median(list_b[low:high])
                            list_g[b] = statistics.median(list_g[low:high])
                            list_r[b] = statistics.median(list_r[low:high])
                    '''
                # print(j)

            res[res > 255] = 255
            res[res < 0] = 0

            res = np.array(res, dtype='uint8')

            panorama[i_piece * size:i_piece * size + res.shape[0],
            j_piece * size:j_piece * size + res.shape[1]] = res
            cv2.imwrite('res06-background-panorama.jpg', panorama)


def part5(panaroma_background, w, h):
    print('part5 started')
    H = np.load('Homography.npy')
    background = cv2.imread(panaroma_background)
    result = cv2.VideoWriter('res07-background-video.mp4', -1, 30, (w, h))
    for i in range(900):
        result.write(cv2.warpPerspective(background, np.linalg.inv(H[i]), (w, h)))
        print(i)
    cv2.destroyAllWindows()
    result.release()


def part6(w,h):
    print('part6 started')
    frames = 900

    H = np.load('Homography.npy')

    # result = cv2.VideoWriter('res05-reference-plane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,(width, height))
    back = imageio.get_writer('vids/background-plane.mp4', fps=30)
    video = cv2.VideoCapture('res07-background-video.mp4')
    for i in range(frames):
        print(i, '-1')
        ret, frame = video.read()
        back.append_data(cv2.cvtColor(cv2.warpPerspective(frame, H[i], (width, height)), cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    back.close()

    back = cv2.VideoCapture('vids/background-plane.mp4')
    video = cv2.VideoCapture('res05-reference-plane.mp4')
    result = imageio.get_writer('res08-foreground-plane.mp4', fps=30)
    lap = imageio.get_writer('laplacian.mp4', fps=30)

    for i in range(frames):
        frame_vid = video.read()[1]
        frame_back = back.read()[1]
        # frame_blur = cv2.GaussianBlur(frame_vid.astype('uint8'), (3, 3), 1).astype('int')
        laplacian = cv2.Laplacian(frame_vid, -1, ksize=5)
        laplacian[laplacian < 50] = 0
        laplacian = cv2.GaussianBlur(laplacian, (15, 15), 5)
        laplacian[laplacian > 10] = 255
        laplacian[laplacian <= 10] = 0

        blur = cv2.GaussianBlur(frame_back, (5, 5), 2)
        fr = np.maximum(frame_vid, blur) - np.minimum(frame_vid, blur)
        fr[fr < 100] = 0
        fr[laplacian != 0] = 0

        fr = cv2.GaussianBlur(fr, (55, 55), 15)
        b1, g1, r1 = cv2.split(fr)
        mask = np.maximum(r1, g1, b1)
        r = frame_vid[:, :, 2]
        r[mask != 0] = 255
        r[r > 255] = 255

        result.append_data(cv2.cvtColor(cv2.warpPerspective(frame_vid, np.linalg.inv(H[i]), (w, h)), cv2.COLOR_BGR2RGB))

        print(i, '-2')
    cv2.destroyAllWindows()
    result.close()
    lap.close()


def part7(backg, w, h):
    video = cv2.VideoCapture(backg)
    im = cv2.imread('res06-background-panorama.jpg')

    video.release()
    cv2.destroyAllWindows()

    video.release()
    cv2.destroyAllWindows()

    H = np.load('Homography.npy')
    result = imageio.get_writer('res09-background-video-wider.mp4', fps=30)

    for i in range(510):
        f1 = cv2.warpPerspective(im, np.linalg.inv(H[i]), (int(w * 1.8), h))
        result.append_data(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
        print(i)
    cv2.destroyAllWindows()
    result.close()


def part8(width, height):
    H = np.load('Homography.npy')
    video = cv2.VideoCapture('res05-reference-plane.mp4')
    result = cv2.VideoWriter('res10-video-shakeless.mp4', -1, 30, (width, height))
    corners_mid = []
    H_mid = []

    for i in range(900):
        print(i, '-1')
        start = max(0, i - 60)
        end = min(i + 60, 900)
        Homog = np.linalg.inv(np.sum(H[start:end], axis=0) / (end - start))
        H_mid.append(Homog)
        Homograph = np.matmul(H[i], Homog)

        c = [np.matmul(Homograph, [0, 0, 1]), np.matmul(Homograph, [0, height, 1]),
             np.matmul(Homograph, [width, 0, 1]), np.matmul(Homograph, [width, height, 1])]
        corners_mid.append([[c[0][0] / c[0][2], c[0][1] / c[0][2]],
                            [c[1][0] / c[1][2], c[1][1] / c[1][2]],
                            [c[2][0] / c[2][2], c[2][1] / c[2][2]],
                            [c[3][0] / c[3][2], c[3][1] / c[3][2]]])

    corners_mid = np.array(corners_mid)
    H_mid = np.array(H_mid)


    for i in range(900):
        print(i, '-2')
        start = 0
        end = 900
        # print(corners_mid[start:end,0,0])
        left_up_corner = [int(np.max(corners_mid[start:end, 0, 0])), int(np.max(corners_mid[start:end, 0, 1]))]
        left_down_corner = [int(np.max(corners_mid[start:end, 1, 0])), int(np.min(corners_mid[start:end, 1, 1]))]
        right_up_corner = [int(np.min(corners_mid[start:end, 2, 0])), int(np.max(corners_mid[start:end, 2, 1]))]
        right_down_corner = [int(np.min(corners_mid[start:end, 3, 0])), int(np.min(corners_mid[start:end, 3, 1]))]

        corners1 = np.array([[0, 0], [0, height], [width, 0], [width, height]])
        corners2 = np.array([left_up_corner, left_down_corner, right_up_corner, right_down_corner])

        Homography = cv2.findHomography(corners2, corners1)
        # print(Homography)
        Homography = np.matmul(Homography[0],H_mid[i])
        frame_vid = video.read()[1]
        result.write(cv2.warpPerspective(frame_vid, Homography, (width, height)))
    cv2.destroyAllWindows()
    video.release()
    result.release()


if __name__ == '__main__':

    key = []
    pan = 'res05-reference-plane.mp4'
    vid = 'video.mp4'
    background = 'res07-background-video.mp4'
    panaroma_background = 'res06-background-panorama.jpg'
    video = cv2.VideoCapture(vid)
    w1 = int(video.get(3))
    h1 = int(video.get(4))

    i = 0
    for i in range(900):
        ret, frame = video.read()
        if i == 29:
            key.append(frame)
        elif i == 89:
            im1 = frame
            key.append(frame)
        elif i == 250:
            f100 = frame
        elif i == 179:
            key.append(frame)
        elif i == 269:
            im2 = frame
            key.append(frame)
        elif i == 359:
            key.append(frame)
        elif i == 449:
            im3 = frame
            key.append(frame)
        elif i == 539:
            key.append(frame)
        elif i == 629:
            key.append(frame)
            im4 = frame
        elif i == 719:
            key.append(frame)
        elif i == 809:
            im5 = frame
            key.append(frame)
        elif i == 869:
            key.append(frame)
        elif i == 0:
            f0 = frame

    video.release()
    cv2.destroyAllWindows()

    video = cv2.VideoCapture(pan)
    width = int(video.get(3))
    height = int(video.get(4))

    SIZE = 400
    t0 = time.time()

    # part1(im2, im3)
    # part2(im1, im2, im3, im4, im5)
    # part3(vid, key)
    # croping(SIZE, pan)
    # part4(SIZE, 7, 15, height, width)
    # part5(panaroma_background, w1, h1)
    part6(w1,h1)
    # part7(background, w1, h1)
    # part8(w1, h1)
    t1 = time.time()
    print('runtime:', int(t1 - t0))
