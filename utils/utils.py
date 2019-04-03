import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def send_msg(msg="...",
             dingding_url="https://oapi.dingtalk.com/robot/send?access_token=774cd9150c43c35e43ec93bc6c91553a5c652417c10fd577bec117ed9f3e3182"
             ):
    '''
    this method is used to send myself a message to remind
    '''
    import requests
    import json
    headers = {"Content-Type": "application/json; charset=utf-8"}

    post_data = {
        "msgtype": "text",
        "text": {
            "content": msg
        }
    }

    requests.post(dingding_url, headers=headers,
                  data=json.dumps(post_data))
