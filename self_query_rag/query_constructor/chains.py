from langchain_openai import OpenAI
from self_query_rag.query_constructor.prompt import *

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class SelfQueryParser(BaseModel):
    query: str = Field(description="text string to compare to document contents")
    filter: dict | str = Field(description="logical condition statements for filtering documents and metadata")
    
def get_examples():
    return EXAMPLE_TEMPLATE.format(attributes=EXAMPLE_ATTRIBUTES, query=EXAMPLE_QUERY, answer=EXAMPLE_ANSWER)
    

def get_results_basic(attributes: str, query:str) -> str:
    """
    Basic model call function, useful for debugging especially because it seems that get_chain version with LangChain fails more often 
    Usage:
    >>> from langchain.output_parsers import PydanticOutputParser
    >>> parser = PydanticOutputParser(pydantic_object=SelfQueryParser)
    >>> parser.invoke(results)
    """
    prompt_prefix = PROMPT_TEMPLATE.format(schema=DEFAULT_SCHEMA, examples=get_examples())
    # add query and data attribute info
    request = REQUEST_TEMPLATE.format(attributes=attributes, query=query)
    final_prompt = prompt_prefix + request
    model = OpenAI()
    return model.predict(final_prompt)


def get_chain():
    """
    This currently fails most of the times at parsing output - still not clear why
    Usage:
    >>> chain = get_chain()
    >>> output = chain.invoke({
        "query": USER_QUERY, "attributes": ATTRIBUTES},
        config={'callbacks': [ConsoleCallbackHandler()]
        })
    """
    model = OpenAI()
    parser = PydanticOutputParser(pydantic_object=SelfQueryParser)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_WITH_FORMATTING + REQUEST_TEMPLATE,
        input_variables=["query", "attributes"],
        partial_variables={
            "schema": DEFAULT_SCHEMA, 
            "examples": get_examples(),
            "format_instructions": parser.get_format_instructions()
        },
    )
    return prompt | model | parser
