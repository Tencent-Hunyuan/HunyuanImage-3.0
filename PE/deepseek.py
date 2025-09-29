# -*- coding: utf-8 -*-
import json
import time
import ast
from loguru import logger
from tencentcloud.common.common_client import CommonClient
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile


class NonStreamResponse(object):
    def __init__(self):
        self.response = ""

    def _deserialize(self, obj):
        self.response = json.dumps(obj)


# DeepSeekClient
class DeepSeekClient(object):
    def __init__(self, key_id, key_secret):
        cred = credential.Credential(key_id, key_secret)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "lkeap.tencentcloudapi.com"
        httpProfile.reqTimeout = 40000  # The streaming interface may take a longer time.
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.common_client = CommonClient("lkeap", "2024-05-22", cred, "ap-guangzhou", profile=clientProfile)

    def run_single_recaption(self, system_prompt, input_prompt):
        post_dict = {
            "Model": "deepseek-r1-0528",
            "Messages": [
                {
                    "Role": "system",
                    "Content": system_prompt
                },
                {
                    "Role": "user",
                    "Content": input_prompt
                }
            ],
            "Stream": False,
        }
        
        while True:
            try:
                resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
                break
            except Exception as e:
                logger.error(e)
                time.sleep(1)
        resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
        response = resp.response
        response = ast.literal_eval(response)
        content = response["Choices"][0]["Message"]["Content"]
        reason = response["Choices"][0]["Message"]["ReasoningContent"]
        return content, reason


if __name__ == "__main__":
     pass