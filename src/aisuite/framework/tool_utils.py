import typing

from langchain_core.tools import BaseTool

SerializedTools = typing.Sequence[typing.Union[typing.Dict[str, typing.Any], type, BaseTool]]