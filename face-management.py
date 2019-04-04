import facePart

import sys
import os


import json
import argparse

def main(args):
    argParser = argparse.ArgumentParser()

    argParser.add_argument("--simple-add", dest="simpleAddPath", type=str)
    argParser.add_argument("--list", dest='listShow', action="store_true")
    argParser.add_argument("--del", dest="deletP", type=str)
    argParser.add_argument("--train", dest='trainGroup', action="store_true")

    argsStore = argParser.parse_args(args)

    faceApp = facePart.FaceApp()


    if (argsStore.simpleAddPath):
        res = faceApp.MSAPI_SimpleAdd(argsStore.simpleAddPath)

        print("%i frames extracted" % res["frameExtracted"])
        print("PersonId: %s" % res["personId"])

        print("FaceIds\n=======")

        for faceId in res["faceIds"]:
            print(faceId)

    elif (argsStore.listShow):
        res = faceApp.MSAPI_GetListGroup()

        print("Persons IDs:")

        for i in range(0, len(res)):
            print(res[i])

    elif (argsStore.deletP):
        print(faceApp.MSAPI_DeletePerson(argsStore.deletP))

    elif argsStore.trainGroup:
        print(faceApp.MSAPI_Train())


if __name__ == "__main__":
    main(sys.argv[1:])
