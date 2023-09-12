from action_recognition.utils import *


class ImagesTransformation:
    """
    This class transforms the input images based on well-defined technique and save the results in a sequence of images having .avi extension:

    - imagesInputPath: the input path from where the class will take the images to be transformed.
    - transformation: the transformation to be applied to the images. The possible choices are: Canny, Sobel XY, Roberts, Binary, No Transformation but images in gray scale, No Transformation. If not provided default will be applied.
    - videosOutputPath: the output path of the sequences. If not provided default will be applied.
    """

    def __init__(self,
                 # the input path of the images, the images must be divided in classes
                 imagesInputPath: str,
                 transformation: str = None,  # the transformation to be applied to the images
                 videosOutputPath: str = None  # the output path of the videos
                 ):
        listOfPossibleTrans = ["Canny", "Sobel_XY",
                               "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]
        if transformation is None:
            print("Transformation assigned by default: No transformation")
            self.__transformation = "No_Trans"  # default option No Transformation
        elif transformation not in listOfPossibleTrans:
            print("Requested transformation is not in the list of the possible transformation: No transformation "
                  "assigned by default")
            self.__transformation = "No_Trans"
        else:
            print("Transforming data by {} transformation".format(transformation))
            self.__transformation = transformation

        assert os.path.exists(imagesInputPath), "Input images path does not exist" # it will contain all the sub-folders for each class
        # find the name of the classes
        classes = imagesInputPath.glob("*")
        self.__classes = list(np.unique([i.name for i in classes]))
        
        self.__globInputImagesPath = imagesInputPath.rglob("*")
        
        if videosOutputPath is None:
            self.__videosTransOutputPath = Path("dataset") / "videos_" + self.__transformation  # default output video folder
        else:
            self.__videosTransOutputPath = videosOutputPath
        # create the output folder if it does not exist
        if not os.path.exists(self.__videosTransOutputPath):
            print("Output folder does not exist: creating a new one")
            os.makedirs(self.__videosTransOutputPath)
            for c in range(len(self.__classes)):
                os.mkdir(os.path.join(self.__videosTransOutputPath, str(c)))
        else:  # delete all the contents of the old folder and create a new one
            print("Output folder already exist: data will be overwritten")
            shutil.rmtree(os.path.join(self.__videosTransOutputPath))
            os.makedirs(self.__videosTransOutputPath)
            for c in range(len(self.__classes)):
                os.mkdir(os.path.join(self.__videosTransOutputPath, str(c)))

        self.__transform_images()  # invoke the function for transforming the images

    def __transform_images(self):
        for fol in self.__globInputImagesPath:  # for each sub-folder
            cl = fol.parent.name
            try:
                # find the index of the corresponding class in the classes list
                indexClass = self.__classes.index(cl)
            except:
                continue
            # find the number of the sub-folder
            objectNum = int(fol.name)
            imgList = os.listdir(fol)  # list of the images in each sub-folder
            imgList.sort()
            imgForVideo = []  # this list collects the transformed images to be transformed in a video
            for im in imgList:
                imPathTmp = os.path.join(fol, im)
                #  Call the method to transform the images
                if self.__transformation == "Canny":
                    imgTrans = CannyTrans(imPathTmp)
                elif self.__transformation == "Sobel_XY":
                    imgTrans = SobelXY(imPathTmp)
                elif self.__transformation == "Binary":
                    imgTrans = Binary(imPathTmp)
                elif self.__transformation == "Roberts":
                    imgTrans = Roberts(imPathTmp)
                elif self.__transformation == "No_Trans_Gray":
                    imgTrans = NoTransGray(imPathTmp)
                else:
                    imgTrans = NoTrans(imPathTmp)
                imgForVideo.append(imgTrans)

            # invoke this function to save the transformed video
            self.__generate_video(imgForVideo, indexClass, objectNum)

    # function that creates the video, the inputs are the list
    def __generate_video(self, frames, classFolder, objectNum):
        # of the images that form the video, the corresponding class, the number of the input images sub-folders
        fps = 30
        height, width, *channels = frames[0].shape

        if objectNum < 10:
            numStr = "000" + str(objectNum)
        elif 10 <= objectNum < 100:
            numStr = "00" + str(objectNum)
        elif 100 <= objectNum < 1000:
            numStr = "0" + str(objectNum)
        else:
            numStr = str(objectNum)
        name = os.path.join(self.__videosTransOutputPath, str(
            classFolder), str(self.__classes[classFolder]) + numStr + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if len(channels) == 0:
            out = cv2.VideoWriter(name, fourcc, fps, (width, height), 0)
        else:
            out = cv2.VideoWriter(name, fourcc, fps, (width, height))
        for i in range(len(frames)):
            tmp = np.uint8(frames[i])
            out.write(tmp)
        out.release()
