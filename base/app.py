from aws_lambda_powertools.event_handler import APIGatewayHttpResolver
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools import Logger

import openai
from openai.error import OpenAIError

app = APIGatewayHttpResolver()
logger = Logger()


@app.get("/get_openai_diagnosis")
def get_openai_diagnosis():
    # Basic checks
    event_body = app.current_event.body
    if not event_body:
        msg = 'No data or payload was sent in the body of the request'
        logger.error(msg)
        return msg, 400

    symptoms = event_body.get("symptoms")
    if not symptoms:
        msg = 'No symptoms were sent in the body of the request'
        logger.error(msg)
        return msg, 400

    # Prepare the question to ask, based on the list of symptoms
    base_question = "Get a medical diagnosis based on these symptoms: "
    question = base_question + base_question + ', '.join(symptoms)

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a medical reference assistant"},
            {"role": "user", "content": question},
        ]
    }

    # Call the OpenAI API and handle possible errors
    try:
        response = openai.ChatCompletion.create(**payload)
    except OpenAIError as e:
        msg = f"OpenAI API request error: {e}"
        logger.error(msg)
        return msg, 502
    except Exception as e:
        msg = f'Server error: {e}'
        logger.error(msg)
        return msg, 500

    # Retrieve Chat GPT's answer
    try:
        gpt_content = response['choices'][0]['message']['content']
    except KeyError as e:
        msg = f'OpenAI API response error: {e}'
        logger.error(msg)
        return msg, 502

    return {"diagnosis": gpt_content}


# Enrich logging with contextual information from Lambda
@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_HTTP)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    return app.resolve(event, context)
