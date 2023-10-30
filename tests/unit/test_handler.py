import json
from unittest.mock import patch

import pytest

from aws_lambda_powertools.utilities.data_classes import APIGatewayProxyEventV2

from base import app


@pytest.fixture()
def apigw_event() -> APIGatewayProxyEventV2:
    """
    Retrieves and returns the API GW Event. This event was generated
    by executing:
    $ sam local generate-event apigateway http-api-proxy

    Returns:
        APIGatewayProxyEventV2: loaded event
    """
    with open("events/event-v2.json") as f:
        return APIGatewayProxyEventV2(json.load(f))


@pytest.fixture()
def lambda_context():
    """
    Creates a valid Context, needed by the lambda handler and resolver.

    Returns:
        LambdaContext: dummy but valid object
    """
    class LambdaContext:
        def __init__(self):
            self.function_name = "test-func"
            self.memory_limit_in_mb = 128
            self.invoked_function_arn = "arn:aws:lambda:eu-west-1:809313241234:function:test-func"
            self.aws_request_id = "52fdfc07-2182-154f-163f-5f0f9a621d72"

        def get_remaining_time_in_millis(self) -> int:
            return 1000

    return LambdaContext()


def mock_openai_object(dummy_content):
    """
    Creates an OpenAI object that will be used for mocking the request and
    response to the actual API.

    Args:
        dummy_content (str): the "answer" from ChatGPT

    Returns:
        OpenAIObject: dummy object
    """
    from openai.openai_object import OpenAIObject

    obj = OpenAIObject()
    message = OpenAIObject()
    content = OpenAIObject()

    content.content = dummy_content
    message.message = content
    obj.choices = [message]

    return obj


def test_lambda_handler(apigw_event, lambda_context):

    with patch("base.app.openai.ChatCompletion.create") as mock_create:
        # Configure the mock to return a successful response
        mock_create.return_value = mock_openai_object('You have a cold!')

        # Call lambda function handler with generated event and context
        test_response = app.lambda_handler(apigw_event, lambda_context)

    print(test_response)

    assert test_response["statusCode"] == 200
    assert "diagnosis" in test_response["body"]


def test_lambda_handler_with_openai_error(apigw_event, lambda_context):
    # Call lambda function handler with generated event and context
    test_response = app.lambda_handler(apigw_event, lambda_context)

    assert test_response["statusCode"] == 502
