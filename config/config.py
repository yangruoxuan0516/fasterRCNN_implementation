class CONFIG:
    
    # RPN

    RPN_ANCHOR_SCALES = [128, 256, 512]
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    RPN_CONV_CHANNELS = 512

    FESTURE_MAP_STRIDE = 16 # after vgg backbone

    RPN_CENTRALIZE_ANCHORS = True

    # RPN_PRE_NMS_TOP_N_TRAIN = 12000
    RPN_NMS_TOP_N_TRAIN = 2000
    # RPN_PRE_NMS_TOP_N_TEST = 6000
    RPN_NMS_TOP_N_TEST = 300

    RPN_NMS_THRESHOLD = 0.7

    RPN_MIN_PROPOSAL_SIZE = 16

    RPN_LOW_THRESHOLD = 0.3
    RPN_HIGH_THRESHOLD = 0.7

    RPN_SAMPLE_POSITIVE_NUM = 128
    RPN_SAMPLE_TOTAL_NUM = 256

    # RPN_LAMBDA = 10


    # ROIhead
    ROI_IN_CHANNELS = RPN_CONV_CHANNELS
    ROI_CLASS_NUM = 21
    ROI_POOL_SIZE = 7
    ROI_FC_CHANNELS = 4096

    ROI_HIGH_THRESHOLD = 0.5
    ROI_LOW_THRESHOLD = 0.1
    ROI_SCORE_THRESHOLD = 0.05

    ROI_MIN_PROPOSAL_SIZE = 16

    ROI_NMS_THRESHOLD = 0.3

    ROI_NMS_TOP_N = 100

    ROI_SAMPLE_POSITIVE_NUM = 32
    ROI_SAMPLE_TOTAL_NUM = 128

    # FRCNN
    FRCNN_IMG_MIN_SIZE = 600
    FRCNN_IMG_MAX_SIZE = 1000

    # TRAIN
    TRAIN_EPOCHS_NUM = 25
    TRAIN_SUB_EPOCHS_NUM = 1


    




config = CONFIG()

# 是不是rpn里class layer的输出要*2？？

# roihead用pretrained vgg16的参数初始化??