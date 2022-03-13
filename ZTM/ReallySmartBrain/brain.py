from imageai.Prediction import ImageClassification
import os
execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsmobilenet_V2()
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)