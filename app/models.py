from pydantic import BaseModel
from code.utils import get_filename

class RequestModel(BaseModel):
    email: str
    face_image_url: str

class ResponseModel(BaseModel):
    acne: float = 0
    acne_image: str = ""
    age_spot: float = 0
    age_spot_image: str = ""
    redness: float = 0
    redness_image: str = ""
    texture: float = 0
    texture_image: str = ""
    wrinkle: float = 0
    wrinkle_image: str = ""
    dent: float = 0
    dent_image: str = ""
    oil: float = 0
    oil_image: str = ""
    eyebag: float = 0
    eyebag_image: str = ""
    eczema: float = 0
    eczema_image: str = ""
    overall_image: str = ""

    @classmethod
    def from_email(cls, email: str):
        return cls(
            acne_image=get_filename(email=email, field="acne"),
            age_spot_image=get_filename(email=email, field="age_spot"),
            redness_image=get_filename(email=email, field="redness"),
            texture_image=get_filename(email=email, field="texture"),
            wrinkle_image=get_filename(email=email, field="wrinkle"),
            dent_image=get_filename(email=email, field="dent"),
            oil_image=get_filename(email=email, field="oil"),
            eyebag_image=get_filename(email=email, field="eyebag"),
            eczema_image=get_filename(email=email, field="eczema"),
            overall_image=get_filename(email=email, field="overall")
        )



class ServerError(BaseModel):
    message: str = "얼굴 정면 모습이 제대로 나온 사진이 필요합니다."

class RequestError(BaseModel):
    message: str = "Error in you request."

class Result(BaseModel):
    score: float
    image: str
    mask: object