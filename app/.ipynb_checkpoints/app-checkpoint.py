from fastapi import FastAPI, UploadFile, File, Response
from .models import RequestModel, ResponseModel, RequestError, ServerError
from code.skin_analysis import skin_analysis
from settings.s3 import MyS3Client

app = FastAPI()

@app.get("/")
def home():
    return {"HealthCheck": "Ok!!!", "ModelVersion": "0.1"}

@app.post("/test", response_model=RequestModel)
async def test_upload(file: UploadFile, email: str):
    s3_client = MyS3Client()
    try:
        folder_path = "pid_" + email + "/testinput/" + file.filename

        face_image_url = s3_client.upload_testfile(file = file.file, folder= folder_path)

        print(face_image_url)

        response = RequestModel(email=email, face_image_url=face_image_url)

        return response
    except Exception as e:
        return Response(ServerError(message=str(e)), status_code=500)

@app.post("/", response_model=ResponseModel, responses={
    400: {"model": RequestError},
    500: {"model": ServerError}
})
def analyze_face(input: RequestModel):
    try: 
        result = skin_analysis(email=input.email, input_data_path=input.face_image_url)
        return result
    except ValueError as e:
        return Response(RequestError(message=str(e)), status_code=400)
    except Exception as e:
        return Response(ServerError(message=str(e)), status_code=500)