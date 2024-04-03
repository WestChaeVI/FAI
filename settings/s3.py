import io
import os
import boto3
from dotenv import load_dotenv
import mimetypes
from io import BytesIO

load_dotenv()

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_bucket_name = os.environ.get("AWS_STORAGE_BUCKET_NAME")

class MyS3Client:

    def __init__(self):
        super(MyS3Client).__init__()
        self.s3_client = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        self.bucket_name = aws_bucket_name
        self.base_path = f"https://{self.bucket_name}.s3.ap-northeast-2.amazonaws.com/media"

    def upload(self, folder: str, file):
        try:
            mime_type, _ = mimetypes.guess_type(folder)
            extra_args = {'ContentType': "image/png"}  # image/jpeg, image/png
            
            file_location = "media/face/" + folder
            
            file_obj = BytesIO()
            file.save(file_obj, format='PNG')
            file_obj.seek(0)
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, file_location, ExtraArgs=extra_args)
            
            return MyS3Client.source_address(folder='/'.join(folder.split('/')[:-1]), file_name=folder.split('/')[-1])
            
        except Exception as e:
            print(e)
            return {"ERROR": e}

    def upload_testfile(self, folder: str, file):
        try:
            file_content = file.read()

            file_location = "media/face/" + folder

            self.s3_client.upload_fileobj(io.BytesIO(file_content), self.bucket_name, file_location, ExtraArgs={'ContentType': "image/png"})
            return MyS3Client.source_address(folder='/'.join(folder.split('/')[:-1]), file_name=folder.split('/')[-1])

        except Exception as e:
            print("#####" + e)
            return {"Error": e}

    @staticmethod
    def source_address(folder: str = None, file_name: str = "user_profile/defaultsilky.png"):
        path = f"https://{aws_bucket_name}.s3.ap-northeast-2.amazonaws.com/media/face"
        if folder:
            path = f"{path}/{folder}"
            
        return f"{path}/{file_name}"
