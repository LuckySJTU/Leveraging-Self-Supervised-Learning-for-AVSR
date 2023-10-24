import cv2 as cv
import dlib
import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm
from tqdm import trange
import os

from multiprocessing import Process, Queue
from collections import deque
from skimage import transform as tf


def get_files(datadir, dataset):
    datalist = []
    for root, dirs, files in os.walk(datadir + "/" + dataset):
        if len(dirs) == 0:
            datalist.extend([root+"/"+filename[:-4] for filename in files if ".mp4" in filename])
    return datalist


def fill_in_data(datalist, flac, png):
    for i, item in enumerate(tqdm(datalist, leave=False, desc="saveh5", ncols=75)):
        # inputAudio, sampFreq = sf.read(item + '.flac')
        # flac[i] = np.array(inputAudio)
        vidInp = cv.imread(item + '.png')
        vidInp = cv.cvtColor(vidInp, cv.COLOR_BGR2GRAY)
        vidInp = cv.imencode(".png", vidInp)[1].tobytes()
        png[i] = np.frombuffer(vidInp, np.uint8)


def main():
    """
        Preparation for model and filelist
    """
    datadir = "/mnt/data/FLLM/voxceleb2"
    PREPROCESSING_NUM_OF_PROCESS = 128
    # print("Searching for files...")
    # filesList = get_files(datadir, 'dev') + get_files(datadir, "test")

    # # check file
    # errorFile = []
    # def check_file(q, filesList):
    #     errorFile=[]
    #     for i in trange(len(filesList)):
    #         obj = cv.VideoCapture(filesList[i]+".mp4")
    #         if not obj.isOpened():
    #             errorFile.append(filesList[i]+"\n")
    #         obj.release()
    #     q.put(errorFile)

    # def splitlist(inlist,chunksize):
    #     return [inlist[x:x+chunksize] for x in range(0,len(inlist),chunksize)]

    # filesListSplitted = splitlist(filesList,int((len(filesList)/PREPROCESSING_NUM_OF_PROCESS)))

    # process_list = []
    # q=Queue()
    # errorFile = []
    # avaiFile = []
    # print(len(filesList))
    # for subFilesList in filesListSplitted:
    #     p=Process(target=check_file,args=(q,subFilesList))
    #     process_list.append(p)
    #     p.Daemon=True
    #     p.start()
    # for p in process_list:
    #     p.join()
    # while not q.empty():
    #     errorFile += q.get()
    # print(len(filesList))
    # for err in errorFile:
    #     filesList.remove(err.strip())
    # avaiFile = [filepath+"\n" for filepath in filesList]

    # print(f"Write file list at {datadir}")
    # with open(datadir+"/errorList3.txt",'w') as f:
    #     f.writelines(errorFile)
    # with open(datadir+"/avaiList3.txt",'w') as f:
    #     f.writelines(avaiFile)
    # assert len(avaiFile)+len(errorFile) == len(filesList)

    with open(datadir+"/avaiList3.txt",'r') as f:
        avaiList = list(map(str.strip, f.readlines()))
    assert len(avaiList) == 1127963
    filesList = avaiList
    for path in tqdm(filesList):
        obj = cv.VideoCapture(path+".mp4")
        if obj.isOpened():
            if os.path.exists(path+".png"):
                continue
            else:
                print(f"No png: {path}")
        else:
            print(f"Cannot open video at: {path}")
        obj.release()

    # filesList = ["/mnt/data/FLLM/voxceleb2/dev/mp4/id02724/H_MYtQtXMo0/00046", "/mnt/data/FLLM/voxceleb2/dev/mp4/id01105/_PmJbaQaVF8/00051"]

    # landmark_detector = dlib.shape_predictor(datadir+"/shape_predictor_68_face_landmarks.dat")
    # mean_face_landmarks = find_mean_face(filesList, PREPROCESSING_NUM_OF_PROCESS, landmark_detector)
    # preprocessing(filesList, PREPROCESSING_NUM_OF_PROCESS, landmark_detector, mean_face_landmarks, False, "lrs")

    # """
    #     Create dataset and Load data
    # """

    # f = h5py.File(datadir+"/voxceleb2.h5", "w")
    # # dt = h5py.vlen_dtype(np.dtype('float32'))
    # # flac = f.create_dataset('flac', (len(filesList),), dtype=dt)
    # dt = h5py.vlen_dtype(np.dtype('uint8'))
    # png = f.create_dataset('png', (len(filesList),), dtype=dt)

    # fill_in_data(filesList, None, png)
    # f.close()



# from find_mean.py
def shape_to_array(shape):
    coords = np.empty((68, 2))
    for i in range(0, 68):
        coords[i][0] = shape.part(i).x
        coords[i][1] = shape.part(i).y
    return coords

def preprocess_sample_fm(file, face_detector, landmark_detector):
    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"

    # for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    landmark_buffer = list()
    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if not len(frame) == 224:
                frame = cv.resize(frame, (224, 224))
            face_rects = face_detector(frame, 0)  # Detect face
            if len(face_rects) < 1:  # No face detected
                continue
            rect = face_rects[0]  # Proper number of face
            landmark = landmark_detector(frame, rect)
            landmark = shape_to_array(landmark)
            landmark_buffer.append(landmark)
        else:
            break
    captureObj.release()

    return np.array(landmark_buffer).sum(0), len(landmark_buffer)

def preprocess_sample_list_fm(filesList, face_detector, landmark_detector, queue):
    sumed_landmarks = np.zeros((68, 2))
    cnt = 0
    for file in tqdm(filesList, leave=True, desc="Face Processing", ncols=75):
        currsumed, currcnt = preprocess_sample_fm(file, face_detector, landmark_detector)
        sumed_landmarks += currsumed
        cnt += currcnt
    ret = queue.get()
    ret['sumed_landmarks'] += sumed_landmarks
    ret['cnt'] += cnt

    queue.put(ret)

def find_mean_face(filesList, processes, landmark_detector):
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    face_detector = dlib.get_frontal_face_detector()

    # multi processing
    queue = Queue()
    queue.put({'sumed_landmarks': np.zeros((68, 2)), 'cnt': 0, 'missed': 0})

    def splitlist(inlist, chunksize):
        return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

    filesListSplitted = splitlist(filesList, int((len(filesList) / processes)))

    process_list = []
    for subFilesList in filesListSplitted:
        p = Process(target=preprocess_sample_list_fm, args=(subFilesList, face_detector, landmark_detector, queue))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()

    return_dict = queue.get()
    return return_dict["sumed_landmarks"] / return_dict["cnt"]


# from preprocessing.py
STD_SIZE = (224, 224)
stablePntsIDs = [33, 36, 39, 42, 45]


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def crop_patch(frames, landmarks, mean_face_landmarks):
    """Crop mouth patch
    :param str frames: video_frames
    :param list landmarks: interpolated landmarks
    """

    for frame_idx, frame in enumerate(frames):
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == 12:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans = tf.estimate_transform('similarity', smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :])
            trans_frame = tf.warp(cur_frame, inverse_map=trans.inverse, output_shape=STD_SIZE)
            trans_frame = trans_frame * 255  # note output from wrap is double image (value range [0,1])
            trans_frame = trans_frame.astype('uint8')
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame, trans_landmarks[48:68], 60, 60))
        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = tf.warp(cur_frame, inverse_map=trans.inverse, output_shape=STD_SIZE)
                trans_frame = trans_frame * 255  # note output from wrap is double image (value range [0,1])
                trans_frame = trans_frame.astype('uint8')
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame, trans_landmarks[48:68], 60, 60))
            return np.array(sequence)
    return None

def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def preprocess_sample(file, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"
    audioFile = file + ".flac"
    roiFile = file + ".png"

    # Extract the audio from the video file using the FFmpeg utility and save it to a flac file.
    if withaudio:
        v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
        os.system(v2aCommand)

    # for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    frames = list()
    landmarks = list()
    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if not len(frame) == 224:
                frame = cv.resize(frame, (224, 224))
            frames.append(frame)

            face_rects = face_detector(frame, 0)  # Detect face
            if len(face_rects) < 1:
                landmarks.append(None)
                continue
            rect = face_rects[0]  # Proper number of face
            landmark = landmark_detector(frame, rect)  # Detect face landmarks
            landmark = shape_to_array(landmark)
            landmarks.append(landmark)
        else:
            break
    captureObj.release()

    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if preprocessed_landmarks is None:
        if defaultcrop == "lrs":
            frames = [frame[52:172, 52:172] for frame in frames]
        else:
            frames = [frame[103: 223, 67: 187] for frame in frames]
    else:
        frames = crop_patch(frames, preprocessed_landmarks, mean_face_landmarks)

    assert frames is not None, "cannot crop from {}.".format(videoFile)

    try:
        cv.imwrite(roiFile, np.concatenate(frames, axis=1).astype(int))
    except:
        print(videoFile)

def preprocess_sample_list(filesList, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop)

def preprocessing(filesList, processes, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")

    face_detector = dlib.get_frontal_face_detector()

    def splitlist(inlist, chunksize):
        return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

    filesListSplitted = splitlist(filesList, int((len(filesList) / processes)))

    process_list = []
    for subFilesList in filesListSplitted:
        p = Process(target=preprocess_sample_list, args=(subFilesList, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()



if __name__ == "__main__":
    main()
