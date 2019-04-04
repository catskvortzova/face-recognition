import json
import cognitive_face as cf
import hashlib
import time
import io
import cv2
class FaceApp:
    def __init__(self):
        self.MSAPI_LoadConf()

    def MSAPI_LoadConf(self):
        with open("faceapi.json") as confJson:
            conf = json.load(confJson)

            cf.BaseUrl.set(conf["serviceUrl"])

            cf.Key.set(conf["key"])

            self.msGroupId = conf["groupId"]

    def MSAPI_SimpleAdd(self, pathToFile, framesCount=5):
        faces = self.__MSAPI_getFrames(pathToFile, framesCount)

        if (faces == 0):
            self.__printError("Video does not contain any face")

        if (self.__MSAPI_isFaces(faces) is None):
            self.__printError("Video does not contain any face")

        try:
            cf.person_group.get(self.msGroupId)
        except cf.util.CognitiveFaceException:
            cf.person_group.create(self.msGroupId)

        names_gen = hashlib.md5(bytes(str(time.time()).encode())).hexdigest()

        personId = cf.person.create(self.msGroupId, names_gen)["personId"]

        faceIds = []
        cf.person_group.update(self.msGroupId,user_data='false')

        for face in faces:
            faceIds.append(cf.person.add_face(io.BytesIO(face), self.msGroupId, personId)["persistedFaceId"])

        return {"frameExtracted": framesCount, "personId": personId, "faceIds": faceIds}

    def MSAPI_GetListGroup(self):
        try:
            listG = cf.person.lists(self.msGroupId)
        except:
            self.__printError('The group does not exist')

        ids = []

        for id in range(0, len(listG)):
            ids.append(listG[id]['personId'])
        if (len(listG) == 0):
            self.__printError('No persons found')

        return ids

    def MSAPI_DeletePerson(self, personId):
        try:
            cf.person_group.get(self.msGroupId)
        except:
            self.__printError("The group does not exist")

        try:
            cf.person.delete(person_group_id=self.msGroupId, person_id=personId)
        except:
            self.__printError("The person does not exist")

        cf.person_group.update(self.msGroupId,user_data='false')
        return "Person deleted"

    def MSAPI_Train(self):
        try:
            cf.person_group.get(self.msGroupId)
        except:
            self.__printError("There is nothing to train")

        if (len(cf.person.lists(self.msGroupId)) == 0):
            self.__printError("There is nothing to train")
        d=cf.person_group.get(person_group_id=self.msGroupId)
        if(d['userData']=='true'):
            self.__printError("Already trained")
        else:
            try:
                cf.person_group.train(self.msGroupId)
            except:
                self.__printError("There is nothing to train")
            cf.person_group.update(person_group_id=self.msGroupId,user_data='true')
            return "Training successfully started"

    def MSAPI_Identify(self, pathToFile, framesCount=5):
        faces = self.__MSAPI_getFrames(pathToFile, framesCount)
        if (faces == 0):
            self.__printError("The video does not follow requirements")
        faceIds = self.__MSAPI_isFaces(faces)

        if (faceIds is None):
            self.__printError("The video does not follow requirements")

        try:
            d=cf.person_group.get(person_group_id=self.msGroupId)
        except:
            self.__printError("The service is not ready")
        if(d['userData']=='false'):
            self.__printError("The service is not ready");

        try:
            results = cf.face.identify(faceIds, person_group_id=self.msGroupId, threshold=0.5)
        except cf.util.CognitiveFaceException:
            self.__printError("The service is not ready");

        candidates = collections.Counter()

        for res in results:
            for candidate in res["candidates"]:
                candidates[candidate["personId"]] += 1

        person = candidates.most_common(1)

        if (len(person) == 0):
            self.__printError("The person was not found");

        if (person[0][1] != framesCount):
            self.__printError("The person was not found");
        return cf.person.get(self.msGroupId, person[0][0])

    def __MSAPI_isFaces(self, imgs):
        faceIds = []

        for img in imgs:
            ids = cf.face.detect(io.BytesIO(img))

            if len(ids) != 1:
                return None

            faceIds.append(ids[0]["faceId"])

        return faceIds

    def __MSAPI_getFrames(self, fileName, framesCount):
        videoStream = cv2.VideoCapture(fileName)

        videoStreamFramesCount = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))

        if (framesCount > videoStreamFramesCount):
            return 0

        frames = []

        jump = videoStreamFramesCount / (framesCount - 1)

        for frame_id in range(1, videoStreamFramesCount + 1):
            if (frame_id % jump < 1 or frame_id == 1):
                videoStream.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                frames.append(cv2.imencode('.jpg', videoStream.read()[1])[1].tostring())

        return frames
    def __printError(self,msg):
        print(msg)
        quit(0)