import sagemaker, boto3, json
from sagemaker.session import Session

from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint

sagemaker_session = Session()
aws_region = boto3.Session().region_name
sm_client = boto3.client("sagemaker", aws_region)
sess = sagemaker.Session()
model_version = "*"

parameters = {
    "max_length": 200,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.95,
    "do_sample": False,
    "temperature": 1,
}


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


content_handler = ContentHandler()

def get_llm_endpoint(llm_endpoint_name) :

    return SagemakerEndpoint(
        endpoint_name=llm_endpoint_name,
        region_name=aws_region,
        model_kwargs=parameters,
        content_handler=content_handler,
    )