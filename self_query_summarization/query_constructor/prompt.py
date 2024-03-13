# Simplified version of 
#   https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_constructor/prompt.py
#   https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/query_constructor/base.py

PROMPT_TEMPLATE = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
{schema}

<< Examples >>
{examples}
"""

PROMPT_TEMPLATE_WITH_FORMATTING = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
{schema}

<< Formatting Instructions >>
{format_instructions}

<< Examples >>
{examples}
"""

DEFAULT_SCHEMA = """\

When responding use a markdown code snippet with a JSON object formatted in the following schema:

```
{
    "query": string \ text string to compare to document contents
    "filter": json \ logical condition statements for filtering documents and metadata
}
```

The query string should match the query provided by the user

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ($eq | $ne | $gt | $gte | $lt | $lte | $in | $nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ($and | $or): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
"""

EXAMPLE_ATTRIBUTES = """ 
{
    "attributes": {
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "genre": {
            "type": "string",
            "description": "The genre of the movie in lowercase. It can contains multiple genres""
        },
        "director": {
            "type": "string",
            "description": "The name of the movie director. It can be multiple names""
        },
        "cast": {
            "type": "string",
            "description": "The name of the actors in the movie""
        },
        "country": {
            "type": "string",
            "description": "The country where the movie was produced""
        },
    }
}
"""

EXAMPLE_QUERY = """ 
Horror movie from the 90s about young people going to vacation on a barren island. One of them, a female artist, has dreams that depict ghastly murders
"""

EXAMPLE_ANSWER = """ 
{
    "query": "Horror movie from the 90s about young people going to vacation on a barren island having nightmares about murders",
    "filter": {
        "$and": [
            {"year": {"$gt": 1990} },
            {"year": {"$lt": 2000} },
            {"genre": {"$in": ["horror"]} }
        ]
    }
}
"""

EXAMPLE_TEMPLATE = """ 
Data Source: 
{attributes}

User Query:
{query}

Structured Request:
{answer}
""" 

REQUEST_TEMPLATE = """
<< Requested output >>
Data Source: 
{attributes}

User Query:
{query}

Structured Request:
Below is the output of the model
"""

def get_prompt_basic(attributes:str, query:str) -> str:
    """
    Basic prompt constructor. 
    After experimenting with an equivalent version using LangChain PromptTemplate I decided to go back to a basic implementation until all 
    problems related to the desired output format are fixed
    
    The equivalent from LangChain would be:
    
    >>> example = EXAMPLE_TEMPLATE.format(attributes=EXAMPLE_ATTRIBUTES, query=EXAMPLE_QUERY, answer=EXAMPLE_ANSWER)
    >>> prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    >>> partial_prompt = prompt.partial(schema=DEFAULT_SCHEMA, examples=example) 
    >>> chain = LLMChain(prompt=partial_prompt, llm=llm)
    >>> results = chain.run(attributes=ATTRIBUTES, query=USER_QUERY)
    >>> results = json.loads(results) 
    """
    # TODO: add possibility to input multiple examples if needed
    example = EXAMPLE_TEMPLATE.format(attributes=EXAMPLE_ATTRIBUTES, query=EXAMPLE_QUERY, answer=EXAMPLE_ANSWER)
    prompt_prefix = PROMPT_TEMPLATE.format(schema=DEFAULT_SCHEMA, examples=example)
    # add query and data attribute info
    request = REQUEST_TEMPLATE.format(attributes=attributes, query=query)
    return prompt_prefix + request

from langchain.prompts import PromptTemplate

def get_prompt_langchain() -> PromptTemplate:
    example = EXAMPLE_TEMPLATE.format(attributes=EXAMPLE_ATTRIBUTES, query=EXAMPLE_QUERY, answer=EXAMPLE_ANSWER)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.partial(schema=DEFAULT_SCHEMA, examples=example)
    
