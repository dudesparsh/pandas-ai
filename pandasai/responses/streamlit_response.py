from typing import Any

import pandas as pd

from pandasai.responses.response_parser import ResponseParser


from PIL import Image

class StreamlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)

    # def format_plot(self, result) -> None:
    #     """
    #     Display plot against a user query in Streamlit
    #     Args:
    #         result (dict): result contains type and value
    #     """
    #     return result["value"]
        
        # Take 1
    def format_plot(self, result) -> None:
    
        # if self._context._config.open_charts:
        #     with Image.open(result["value"]) as img:
        #         img.show()

        return result["value"]


    def format_dataframe(self, result: dict) -> pd.DataFrame:
        """
        Format dataframe generate against a user query
        Args:
            result (dict): result contains type and value
        Returns:
            Any: Returns depending on the user input
        """
        return result["value"]

    def format_other(self, result) -> Any:
        """
        Format other results
        Args:
            result (dict): result contains type and value
        Returns:
            Any: Returns depending on the user input
        """
        return result["value"]
