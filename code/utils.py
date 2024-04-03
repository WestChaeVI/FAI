import uuid
import datetime

def get_filename(email: str, field: str) :
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day

    filename = 'pid_{}/{}/{}/{}/image{}.{}.png'.format(email, year, month, day, field, str(uuid.uuid4()))
    
    return filename
