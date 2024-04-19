import json
import cv2
import requests
import os
import names
import shutil
from openai import OpenAI

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    max_confidence = 0
    best_face = None
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold and confidence > max_confidence:
            max_confidence = confidence
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            best_face = [x1,y1,x2,y2]
    if best_face is not None:
        faceBoxes.append(best_face)
        cv2.rectangle(frameOpencvDnn, (best_face[0],best_face[1]), (best_face[2],best_face[3]), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def detect_age_gender(image_path):
    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    frame=cv2.imread(image_path)
    padding=20
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    results = []
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        results.append({"age": age, "gender": gender})

    return results

def get_fake_person_image(image_path):
    url = "https://thispersondoesnotexist.com"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(image_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Unable to download image. Status code:", response.status_code)

def generate_name(gender):
    if gender.lower() == 'male':
        return names.get_first_name(gender='male')
    elif gender.lower() == 'female':
        return names.get_first_name(gender='female')
    else:
        return names.get_first_name()

def make_random_person():
    image_path = "person.jpg"
    get_fake_person_image(image_path)
    results = detect_age_gender(image_path)
    for result in results:
        name = generate_name(result['gender'])

        print(f"Creating random person... Name: {name}, Age: {result['age']}, Gender: {result['gender']}")

        # Create a directory for the person
        new_dir = os.path.join('people', name)
        os.makedirs(new_dir, exist_ok=True)
        new_image_path = os.path.join(new_dir, f'{name}.jpg')
        shutil.copy(image_path, new_image_path)

        # Generate a story about the person
        story = generate_story(name, result['gender'], result['age'])

        data = {
            'name': name,
            'age': result['age'],
            'gender': result['gender'],
            'image_url': new_image_path,
            'story': story
        }

        with open(os.path.join(new_dir, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=4)

def get_openai_key():
    return os.getenv('RANDOM_PEOPLE_OPENAI_KEY')

def generate_story(name, gender, age):
    prompt = f"{name} is a {age} year old {gender}. "

    try:
        with open('prompt.txt', 'r') as file:
            custom_prompt = file.read()
            prompt += custom_prompt
    except FileNotFoundError:
            print("No custom prompt found. Using default prompt. To use a custom prompt, create a file named 'prompt.txt' in the current directory with your custom prompt.")
            prompt += "create a story about them using the present tense."
    
    client = OpenAI(
        # This is the default and can be omitted
        api_key=get_openai_key()
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
     # Extract the content of the assistant's message
    story = chat_completion.choices[0].message.content
    return story

if __name__ == "__main__":
    make_random_person()

