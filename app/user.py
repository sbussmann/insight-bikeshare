import datetime as dt
import re


class User:

    """

    Each user gets their own "growdir".  Ensures that if multiple users access
    the app simultaneously, bad things do not happen.

    """

    def __init__(self, uid):
        self.last_activity_time = dt.datetime.today()

    def record_as_active(self):
        self.last_activity_time = dt.datetime.today()

def get_next_user_id():
    f = open('static/users.stat', 'r')
    match = re.search(r'total number of users = (\d+)', f.read())
    if match:
        next_user_id = int(match.group(1))
    else:
        next_user_id = 0
    f.close()
    return next_user_id

def put_next_user_id(next_id):
    f = open('static/users.stat', 'w')
    f.write('total number of users = ' + str(next_id))
    f.close()
