#
# def compute_rgb(video_path, sample_rate):
#     """Compute RGB"""
#     rgb = []
#     vidcap = cv2.VideoCapture(str(video_path))
#     success, frame = vidcap.read()
#     counter = 0
#     while success:
#         frame = cv2.resize(frame, (342, 256))
#         if counter%sample_rate==0:
#             frame = (frame / 255.) # * 2 - 1
#             frame = frame[16:240, 59:283]
#             rgb.append(frame)
#         success, frame = vidcap.read()
#     vidcap.release()
#     rgb = rgb[:-1]
#     rgb = np.asarray([np.array(rgb)])
#     log('save rgb with shape ', rgb.shape)
#     np.save(SAVE_DIR / 'rgb.npy', rgb)
#     return rgb
#
#
# def compute_TVL1(video_path):
#     """Compute the TV-L1 optical flow."""
#     flow = []
#     TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
#     vidcap = cv2.VideoCapture(str(video_path))
#     success, frame1 = vidcap.read()
#     frame1 = cv2.resize(frame1, (342, 256))
#     bins = np.linspace(-20, 20, num=256)
#     prev = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
#     vid_len = get_video_length(video_path)
#     for _ in range(0, vid_len - 1):
#         success, frame2 = vidcap.read()
#         frame2 = cv2.resize(frame2, (342, 256))
#         curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
#         curr_flow = TVL1.calc(prev, curr, None)
#         assert (curr_flow.dtype == np.float32)
#
#         # Truncate large motions
#         curr_flow[curr_flow >= 20] = 20
#         curr_flow[curr_flow <= -20] = -20
#
#         # digitize and scale to [-1;1]
#         curr_flow = np.digitize(curr_flow, bins)
#         curr_flow = (curr_flow / 255.) * 2 - 1
#
#         # cropping the center
#         print(curr_flow.shape)
#         curr_flow = curr_flow[16:240, 59:283]
#         flow.append(curr_flow)
#         prev = curr
#     vidcap.release()
#     flow = np.asarray([np.array(flow)])
#     log('Save flow with shape ', flow.shape)
#     np.save(SAVE_DIR / 'flow.npy', flow)
#     return flow


# def get_video_length(video_path):
#     video_path = Path(video_path)
#     if video_path.suffix not in _VIDEO_EXT:
#         raise ValueError('Extension "%s" not supported' % video_path.suffix)
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise ValueError("Could not open the file.\n{}".format(video_path))
#     if cv2.__version__ >= '3.0.0':
#         cap_prop_frame_count = cv2.CAP_PROP_FRAME_COUNT
#     else:
#         cap_prop_frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT
#     length = int(cap.get(cap_prop_frame_count))
#     cap.release()
#     return length